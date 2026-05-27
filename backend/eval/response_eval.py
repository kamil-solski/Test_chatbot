"""End-to-end evaluation: LLM-as-a-judge scoring of full pipeline responses.

Dimensions (each 1-5):
  correctness   — factual accuracy against reference or conversation history
  coherence     — relevance, structure, clarity
  groundedness  — response is grounded in retrieved context, not hallucinated

Usage:
    from eval.e2e_eval import evaluate_responses, E2EReport

    samples = [
        {
            "question": "What is my favorite language?",
            "response": "Your favorite language is Python.",
            "reference": "Python",          # optional gold answer
            "context": "User said: My favorite language is Python.",  # optional
        },
        ...
    ]
    report = await evaluate_responses(llm, samples)
"""
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage


@dataclass
class E2EReport:
    correctness_mean: float    # 1-5
    coherence_mean: float      # 1-5
    groundedness_mean: float   # 1-5
    samples: list[dict]


async def evaluate_responses(
    llm: BaseChatModel,
    samples: list[dict],
) -> E2EReport:
    """Evaluate a list of (question, response) samples.

    Each sample dict:
        question  (str)  — the user query
        response  (str)  — the model's response to judge
        reference (str)  — optional gold-standard answer
        context   (str)  — optional retrieved context injected into the model
    """
    results = []
    for sample in samples:
        scores = await _judge(
            llm,
            question=sample["question"],
            response=sample["response"],
            reference=sample.get("reference", ""),
            context=sample.get("context", ""),
        )
        results.append({"question": sample["question"], **scores})

    def _mean(key: str) -> float:
        vals = [r[key] for r in results if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    return E2EReport(
        correctness_mean=_mean("correctness"),
        coherence_mean=_mean("coherence"),
        groundedness_mean=_mean("groundedness"),
        samples=results,
    )


async def _judge(
    llm: BaseChatModel,
    question: str,
    response: str,
    reference: str,
    context: str,
) -> dict:
    ref_block = f"\nReference answer:\n{reference}" if reference else ""
    ctx_block = f"\nContext available to the model:\n{context}" if context else ""

    prompt = f"""You are an objective evaluator of AI assistant responses.

Question: {question}{ref_block}{ctx_block}

Model response:
{response}

Rate the response on THREE dimensions, each 1-5:

CORRECTNESS: Is the response factually accurate?
  1=Factually wrong  3=Partially correct  5=Fully correct

COHERENCE: Is the response relevant, structured, and easy to follow?
  1=Incoherent  3=Somewhat clear  5=Excellent

GROUNDEDNESS: Is the response grounded in context or established facts (not hallucinated)?
  1=Clearly hallucinated  3=Mixed  5=Fully grounded

Reply in EXACTLY this format — integers only:
CORRECTNESS: <1-5>
COHERENCE: <1-5>
GROUNDEDNESS: <1-5>"""

    raw = (await llm.ainvoke([HumanMessage(content=prompt)])).content.strip()
    return _parse(raw)


def _parse(text: str) -> dict:
    scores: dict[str, float | None] = {"correctness": None, "coherence": None, "groundedness": None}
    for line in text.splitlines():
        upper = line.strip().upper()
        for key in scores:
            if upper.startswith(key.upper() + ":"):
                try:
                    scores[key] = float(upper.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
    return {k: v if v is not None else 3.0 for k, v in scores.items()}
