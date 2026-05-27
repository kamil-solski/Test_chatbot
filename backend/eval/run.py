"""CLI runner for the evaluation module.

Run from the backend/ directory:
    python -m eval.run                        # router only (no API key needed)
    python -m eval.run -c summarizer          # summarizer (needs OPENAI_API_KEY)
    python -m eval.run -c retrieval           # retrieval  (needs OPENAI_API_KEY)
    python -m eval.run -c e2e                 # e2e notes  (shows usage instructions)
    python -m eval.run -c all                 # router + summarizer + retrieval
"""
import argparse
import asyncio

from config import CONFIG, LITELLM_BASE_URL, OPENAI_API_KEY


def _make_llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=CONFIG["eval"]["judge_model"],
        api_key=OPENAI_API_KEY,
        base_url=LITELLM_BASE_URL,
        temperature=0.0,
    )


# --------------------------------------------------------------------------- #
# Router                                                                        #
# --------------------------------------------------------------------------- #

def run_router() -> None:
    from eval.router_eval import evaluate_router

    # No available_topics — matches a fresh chat session before any topic
    # cluster has formed. Case #16 will misclassify as SEMANTIC_RECALL because
    # of this; see README "Router known under-classifications".
    report = evaluate_router()

    print("\n=== Router Evaluation ===")
    print(f"Accuracy: {report.correct}/{report.total}  ({report.accuracy:.1%})")

    if report.failures:
        print(f"\nFailures ({len(report.failures)}):")
        for f in report.failures:
            print(f"  #{f['id']:>2}  {f['prompt'][:65]!r}")
            print(f"        expected={f['expected']}  predicted={f['predicted']}")
            if f["notes"]:
                print(f"        note: {f['notes']}")

    print("\nPer-class metrics:")
    for cls, m in sorted(report.per_class.items()):
        print(f"  {cls:<20}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")

    print("\nConfusion matrix (expected → predicted):")
    all_intents = sorted({k for row in report.confusion.values() for k in row} | set(report.confusion))
    header = f"{'':20}" + "".join(f"{c:>18}" for c in all_intents)
    print("  " + header)
    for expected in all_intents:
        row = report.confusion.get(expected, {})
        cells = "".join(f"{row.get(p, 0):>18}" for p in all_intents)
        print(f"  {expected:<20}{cells}")


# --------------------------------------------------------------------------- #
# Summarizer                                                                    #
# --------------------------------------------------------------------------- #

async def run_summarizer() -> None:
    from eval.summarizer_eval import evaluate_live
    from eval.dataset import SAMPLE_CONVERSATION

    llm = _make_llm()
    print("\n=== Summarizer Evaluation ===")
    print(f"Conversation: {len(SAMPLE_CONVERSATION)} messages, chunk_size=4, window_size=2")

    report = await evaluate_live(llm, SAMPLE_CONVERSATION, chunk_size=4, window_size=2)

    if not report.samples:
        print("No chunks produced — conversation too short for given chunk_size/window_size.")
        return

    print(f"ROUGE-L mean:      {report.rouge_l_mean if report.rouge_l_mean is not None else 'n/a (install rouge-score)'}")
    print(f"Faithfulness mean: {report.faithfulness_mean:.2f} / 5.0")
    for i, s in enumerate(report.samples, 1):
        print(f"  Chunk {i}: ROUGE-L={s['rouge_l']}  Faithfulness={s['faithfulness']}")


# --------------------------------------------------------------------------- #
# Retrieval                                                                     #
# --------------------------------------------------------------------------- #

async def run_retrieval() -> None:
    from eval.retrieval_eval import evaluate_retrieval
    from eval.dataset import SAMPLE_CONVERSATION
    from rag.store import SessionRAGStore

    llm = _make_llm()
    store = SessionRAGStore()

    print("\n=== Retrieval Evaluation ===")
    print("Indexing sample conversation into SessionRAGStore...")
    store.index_messages("_eval", SAMPLE_CONVERSATION)

    queries = [
        "What is prompt engineering?",
        "What is my favorite programming language?",
    ]

    from langchain_core.documents import Document

    samples = []
    all_docs: list[Document] = store.retrieve("_eval", "anything", k=len(SAMPLE_CONVERSATION))
    for q in queries:
        retrieved = store.retrieve("_eval", q, k=4)
        samples.append({"query": q, "retrieved": retrieved, "all_docs": all_docs})

    report = await evaluate_retrieval(llm, samples)

    print(f"Context Precision mean: {report.precision_mean:.3f}")
    if report.recall_mean is not None:
        print(f"Context Recall mean:    {report.recall_mean:.3f}")
    for s in report.samples:
        recall_str = f"  Recall={s['recall']}" if s["recall"] is not None else ""
        print(f"  {s['query']!r}")
        print(f"    Retrieved={s['retrieved_count']}  Relevant={s['relevant_retrieved']}  Precision={s['precision']}{recall_str}")


# --------------------------------------------------------------------------- #
# E2E                                                                           #
# --------------------------------------------------------------------------- #

def run_e2e_help() -> None:
    print("""
=== E2E Evaluation ===
E2E evaluation requires recorded (question, response) pairs from a live session.

Steps:
  1. Run the 18 prompts from eval/dataset.py through /api/chat with the desired
     history.strategy (one chat session per strategy you want to compare).
  2. Capture each (question, response) pair from the API response.
  3. Call evaluate_responses() with the recorded samples:

    from eval import evaluate_responses
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    samples = [
        {
            "question": "What was the first topic we discussed?",
            "response": "<captured model response>",
            "reference": "Prompt engineering",
            "context": "<context injected by the strategy>",
        },
        ...
    ]
    report = await evaluate_responses(llm, samples)
    print(report)

See eval/dataset.py -> E2E_REFERENCES for reference answers to the 18-prompt test set.
""")


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Eval runner — run from backend/ directory")
    parser.add_argument(
        "-c", "--component",
        choices=["router", "summarizer", "retrieval", "e2e", "all"],
        default="router",
        help="Component to evaluate (default: router, the only one that needs no API key)",
    )
    args = parser.parse_args()

    if args.component in ("router", "all"):
        run_router()

    if args.component in ("summarizer", "all"):
        asyncio.run(run_summarizer())

    if args.component in ("retrieval", "all"):
        asyncio.run(run_retrieval())

    if args.component == "e2e":
        run_e2e_help()


if __name__ == "__main__":
    main()
