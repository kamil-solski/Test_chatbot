"""Router classification accuracy — synchronous, no LLM required."""
from collections import defaultdict
from dataclasses import dataclass, field

from chat_history.router import classify_query

from .dataset import ROUTER_CASES, RouterCase


@dataclass
class RouterReport:
    accuracy: float
    correct: int
    total: int
    per_class: dict[str, dict]  # intent_value -> {tp, fp, fn, precision, recall, f1}
    failures: list[dict]        # cases where predicted != expected
    confusion: dict[str, dict]  # expected_value -> predicted_value -> count


def evaluate_router(
    cases: list[RouterCase] | None = None,
    available_topics: list[str] | None = None,
) -> RouterReport:
    """Run the heuristic router on all labeled cases and return a RouterReport.

    Args:
        cases: Defaults to the 18-prompt ROUTER_CASES from dataset.py.
        available_topics: Topic names to pass to classify_query (simulates a session
            that has already seen those topics). Relevant for TOPIC_RECALL cases.
    """
    if cases is None:
        cases = ROUTER_CASES

    correct = 0
    failures: list[dict] = []
    confusion: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    raw: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for case in cases:
        decision = classify_query(case.prompt, available_topics)
        predicted = decision.intent
        expected = case.expected_intent

        confusion[expected.value][predicted.value] += 1

        if predicted == expected:
            correct += 1
            raw[expected.value]["tp"] += 1
        else:
            raw[expected.value]["fn"] += 1
            raw[predicted.value]["fp"] += 1
            failures.append({
                "id": case.id,
                "prompt": case.prompt,
                "expected": expected.value,
                "predicted": predicted.value,
                "router_reason": decision.reason,
                "notes": case.notes,
            })

    per_class: dict[str, dict] = {}
    for intent_val, counts in raw.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[intent_val] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    return RouterReport(
        accuracy=round(correct / len(cases), 3) if cases else 0.0,
        correct=correct,
        total=len(cases),
        per_class=per_class,
        failures=failures,
        confusion=dict({k: dict(v) for k, v in confusion.items()}),
    )
