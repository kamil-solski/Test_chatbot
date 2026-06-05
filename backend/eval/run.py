"""Evaluation runner — evaluates a live chat session stored in Postgres.

Run from inside the backend container:

    docker compose exec backend python -m eval.run --session <uuid>
    docker compose exec backend python -m eval.run --session <uuid> -c router
    docker compose exec backend python -m eval.run --session <uuid> -c all

The runner reads the session's turns and chunk summaries from Postgres, fires
the relevant judges, prints a summary, and appends an eval_runs row to the DB
for historical comparison.
"""
import argparse
import asyncio

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

import db
from chat_history.router import Intent, classify_query
from config import CONFIG, LITELLM_BASE_URL, OPENAI_API_KEY
from eval.e2e_eval import evaluate_responses
from eval.retrieval_eval import evaluate_retrieval
from eval.router_eval import RouterCase, evaluate_router
from eval.summarizer_eval import evaluate_summarizer

VALID_INTENTS = {i.value for i in Intent}


def _make_llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=CONFIG["eval"]["judge_model"],
        api_key=OPENAI_API_KEY,
        base_url=LITELLM_BASE_URL,
        temperature=0.0,
    )


# --------------------------------------------------------------------------- #
# Router (uses LLM judge to label unlabeled queries)                            #
# --------------------------------------------------------------------------- #

async def _judge_intent(llm, query: str) -> str:
    prompt = f"""Classify this query into exactly one of these intent labels:
- continuation  (short follow-up; the previous turn carries the context)
- normal        (standalone new question, no recall of prior turns needed)
- position      (asks about the first/last/Nth previous prompt or topic)
- topic_recall  (refers back to a specific named topic discussed earlier)
- semantic_recall (asks about something said earlier without naming a specific topic)

Reply with ONLY the label, nothing else.

Query: {query}"""
    response = await llm.ainvoke(prompt)
    label = response.content.strip().lower().split()[0] if response.content.strip() else ""
    return label if label in VALID_INTENTS else "normal"


async def run_router_eval(session_id: str, turns: list[dict], llm) -> dict:
    """Evaluate the router on session turns.

    y_true comes from: intent_user_override (if set) → intent_judge_label
    (cached LLM label) → fresh LLM judge call (then cached back to DB).
    """
    cases: list[RouterCase] = []
    per_turn = []

    for t in turns:
        question = t["question"]
        routing = t.get("routing") or {}
        intent_predicted_str = routing.get("intent")

        # Resolve y_true
        y_true = t.get("intent_user_override") or t.get("intent_judge_label")
        if not y_true:
            y_true = await _judge_intent(llm, question)
            await db.update_turn_judge_label(t["id"], y_true)

        try:
            expected = Intent(y_true)
        except ValueError:
            continue  # judge returned something unrecognized; skip this turn

        cases.append(RouterCase(
            id=t["turn_number"],
            prompt=question,
            expected_intent=expected,
        ))
        per_turn.append({
            "turn_number": t["turn_number"],
            "intent_predicted": intent_predicted_str,
            "intent_y_true": y_true,
            "agree": intent_predicted_str == y_true,
        })

    if not cases:
        return {"summary": {"router_n": 0}, "per_turn": per_turn}

    report = evaluate_router(cases=cases)
    return {
        "summary": {
            "router_accuracy": report.accuracy,
            "router_n": report.total,
            "router_per_class": report.per_class,
        },
        "per_turn": per_turn,
    }


# --------------------------------------------------------------------------- #
# Retrieval (only meaningful for turns that actually retrieved docs)            #
# --------------------------------------------------------------------------- #

async def run_retrieval_eval(turns: list[dict], llm) -> dict:
    samples = []
    sample_to_turn = []

    # Build a session-wide pool of all retrieved docs for recall computation
    all_seen: list[dict] = []
    seen_keys: set[tuple[str, str]] = set()
    for t in turns:
        for doc in t.get("retrieved") or []:
            key = (doc.get("role", "human"), doc.get("content", ""))
            if key not in seen_keys:
                seen_keys.add(key)
                all_seen.append(doc)
    all_docs = [Document(page_content=d["content"], metadata={"role": d.get("role", "human")}) for d in all_seen]

    for t in turns:
        retrieved_raw = t.get("retrieved") or []
        if not retrieved_raw:
            continue
        retrieved_docs = [
            Document(page_content=d["content"], metadata={"role": d.get("role", "human")})
            for d in retrieved_raw
        ]
        samples.append({
            "query": t["question"],
            "retrieved": retrieved_docs,
            "all_docs": all_docs,
        })
        sample_to_turn.append(t["turn_number"])

    if not samples:
        return {"summary": {"retrieval_n": 0}, "per_turn": []}

    report = await evaluate_retrieval(llm, samples)
    per_turn = [
        {
            "turn_number": sample_to_turn[i],
            "retrieval_precision": s["precision"],
            "retrieval_recall": s["recall"],
            "retrieved_count": s["retrieved_count"],
            "relevant_retrieved": s["relevant_retrieved"],
        }
        for i, s in enumerate(report.samples)
    ]
    return {
        "summary": {
            "retrieval_precision_mean": report.precision_mean,
            "retrieval_recall_mean": report.recall_mean,
            "retrieval_n": len(samples),
        },
        "per_turn": per_turn,
    }


# --------------------------------------------------------------------------- #
# Summarizer (only meaningful for smart strategy with chunk summaries)          #
# --------------------------------------------------------------------------- #

async def run_summarizer_eval(session_id: str, llm) -> dict:
    chunks = await db.fetch_session_chunks(session_id)
    if not chunks:
        return {"summary": {"summarizer_n": 0}, "per_chunk": []}

    samples = []
    for c in chunks:
        msgs = []
        for m in c["source_messages"]:
            cls = HumanMessage if m.get("role") == "human" else AIMessage
            msgs.append(cls(content=m.get("content", "")))
        samples.append((msgs, c["summary_text"]))

    report = await evaluate_summarizer(llm, samples)
    per_chunk = [
        {
            "chunk_index": chunks[i]["chunk_index"],
            "rouge_l": s["rouge_l"],
            "faithfulness": s["faithfulness"],
        }
        for i, s in enumerate(report.samples)
    ]
    return {
        "summary": {
            "summarizer_rouge_l_mean": report.rouge_l_mean,
            "summarizer_faithfulness_mean": report.faithfulness_mean,
            "summarizer_n": len(report.samples),
        },
        "per_chunk": per_chunk,
    }


# --------------------------------------------------------------------------- #
# E2E (judges each response against optional reference + context)               #
# --------------------------------------------------------------------------- #

async def run_e2e_eval(turns: list[dict], llm) -> dict:
    samples = []
    sample_to_turn = []
    for t in turns:
        samples.append({
            "question": t["question"],
            "response": t["response"],
            "reference": t.get("reference_answer") or "",
            "context": t.get("context_injected") or "",
        })
        sample_to_turn.append(t["turn_number"])

    if not samples:
        return {"summary": {"e2e_n": 0}, "per_turn": []}

    report = await evaluate_responses(llm, samples)
    per_turn = [
        {
            "turn_number": sample_to_turn[i],
            "e2e_correctness": s.get("correctness"),
            "e2e_coherence": s.get("coherence"),
            "e2e_groundedness": s.get("groundedness"),
        }
        for i, s in enumerate(report.samples)
    ]
    return {
        "summary": {
            "e2e_correctness_mean": report.correctness_mean,
            "e2e_coherence_mean": report.coherence_mean,
            "e2e_groundedness_mean": report.groundedness_mean,
            "e2e_n": len(samples),
        },
        "per_turn": per_turn,
    }


# --------------------------------------------------------------------------- #
# Orchestration                                                                 #
# --------------------------------------------------------------------------- #

async def run_session_eval(session_id: str, component: str) -> None:
    await db.init_pool()
    try:
        turns = await db.fetch_session_turns(session_id)
        if not turns:
            print(f"No turns found for session {session_id}.")
            print("Chat in the GUI first, then re-run.")
            return

        llm = _make_llm()
        aggregate: dict = {}
        per_turn_by_number: dict[int, dict] = {t["turn_number"]: {"turn_number": t["turn_number"]} for t in turns}
        per_chunk_results: list[dict] = []

        if component in ("router", "all"):
            r = await run_router_eval(session_id, turns, llm)
            aggregate.update(r["summary"])
            for pt in r["per_turn"]:
                per_turn_by_number.setdefault(pt["turn_number"], {"turn_number": pt["turn_number"]}).update(pt)

        if component in ("retrieval", "all"):
            r = await run_retrieval_eval(turns, llm)
            aggregate.update(r["summary"])
            for pt in r["per_turn"]:
                per_turn_by_number.setdefault(pt["turn_number"], {"turn_number": pt["turn_number"]}).update(pt)

        if component in ("summarizer", "all"):
            r = await run_summarizer_eval(session_id, llm)
            aggregate.update(r["summary"])
            per_chunk_results = r.get("per_chunk", [])

        if component in ("e2e", "all"):
            r = await run_e2e_eval(turns, llm)
            aggregate.update(r["summary"])
            for pt in r["per_turn"]:
                per_turn_by_number.setdefault(pt["turn_number"], {"turn_number": pt["turn_number"]}).update(pt)

        per_turn_results = sorted(per_turn_by_number.values(), key=lambda x: x["turn_number"])
        _print_report(session_id, len(turns), aggregate, per_turn_results, per_chunk_results)

        await db.insert_eval_run(
            session_id=session_id,
            judge_model=CONFIG["eval"]["judge_model"],
            turns_evaluated=len(turns),
            aggregate_metrics=aggregate,
            per_turn_results=per_turn_results,
        )
        print(f"\nResults saved to eval_runs (session_id={session_id}).")
    finally:
        await db.close_pool()


def _print_report(session_id, n_turns, aggregate, per_turn, per_chunk):
    print("\n=== Evaluation Session Summary ===")
    print(f"Session: {session_id}")
    print(f"Turns evaluated: {n_turns}")
    print(f"Judge model: {CONFIG['eval']['judge_model']}")

    print("\n--- Aggregate metrics ---")
    for k in sorted(aggregate.keys()):
        v = aggregate[k]
        if isinstance(v, dict):
            print(f"  {k}:")
            for sub_k, sub_v in v.items():
                print(f"    {sub_k}: {sub_v}")
        else:
            print(f"  {k}: {v}")

    if per_turn:
        print("\n--- Per-turn ---")
        for pt in per_turn:
            line = f"  #{pt['turn_number']:>2}  "
            extras = []
            if "agree" in pt:
                ok = "OK" if pt["agree"] else "MISS"
                extras.append(f"router={pt.get('intent_predicted')}->{pt.get('intent_y_true')} [{ok}]")
            if "retrieval_precision" in pt:
                extras.append(f"P={pt['retrieval_precision']} R={pt['retrieval_recall']}")
            if "e2e_correctness" in pt:
                extras.append(
                    f"e2e={pt.get('e2e_correctness')}/{pt.get('e2e_coherence')}/{pt.get('e2e_groundedness')}"
                )
            print(line + " | ".join(extras))

    if per_chunk:
        print("\n--- Per-chunk (summarizer) ---")
        for pc in per_chunk:
            print(f"  chunk {pc['chunk_index']}  ROUGE-L={pc['rouge_l']}  faithfulness={pc['faithfulness']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval runner — evaluates a live session from Postgres")
    parser.add_argument("--session", required=True, help="Session UUID to evaluate")
    parser.add_argument(
        "-c", "--component",
        choices=["router", "summarizer", "retrieval", "e2e", "all"],
        default="all",
        help="Component to evaluate (default: all)",
    )
    args = parser.parse_args()
    asyncio.run(run_session_eval(args.session, args.component))


if __name__ == "__main__":
    main()
