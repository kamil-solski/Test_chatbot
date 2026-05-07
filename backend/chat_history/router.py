"""Heuristic intent classifier — decides which retrieval pattern fits a query."""
import re
from dataclasses import dataclass
from enum import Enum


class Intent(str, Enum):
    CONTINUATION = "continuation"        # short follow-up — window only
    POSITION = "position"                # "first/last/Nth prompt" — raw log
    TOPIC_RECALL = "topic_recall"        # mentions a known topic — filter→vector
    SEMANTIC_RECALL = "semantic_recall"  # "remember when..." — pure vector
    NORMAL = "normal"                    # default — window + summary


@dataclass
class RoutingDecision:
    intent: Intent
    use_vector: bool = False
    use_raw_log: bool = False
    topic_filter: str | None = None
    reason: str = ""


_POSITION_PATTERNS = [
    re.compile(r"\b(first|initial|original|starting)\s+(prompt|message|question|topic|subject|thing|discussion)\b"),
    re.compile(r"\b(last|previous|prior)\s+(prompt|message|question|topic|subject|thing|discussion)\b"),
    re.compile(r"\b(\d+)(st|nd|rd|th)?\s+(prompt|message|question)\b"),
    re.compile(r"\bwhat\s+(was|did)\s+i\s+(ask|say)\s+(first|initially|originally)\b"),
]

_RECALL_PATTERNS = [
    re.compile(r"\bremember\s+(when|that|how)\b"),
    re.compile(r"\b(earlier|previously|before)\s+(we|you|i)\b"),
    re.compile(r"\bgoing\s+back\s+to\b"),
    re.compile(r"\bwe\s+(discussed|talked\s+about|covered|mentioned)\b"),
    re.compile(r"\bas\s+i\s+said\s+(earlier|before)\b"),
]


def classify_query(text: str, available_topics: list[str] | None = None) -> RoutingDecision:
    """Classify a query into a retrieval intent. Heuristic-based, no LLM call."""
    lower = text.lower().strip()

    # Continuation: very short queries → window is enough
    if len(lower.split()) < 4:
        return RoutingDecision(
            intent=Intent.CONTINUATION,
            reason="short query — window only",
        )

    # Position queries → deterministic raw-log lookup
    for pat in _POSITION_PATTERNS:
        if pat.search(lower):
            return RoutingDecision(
                intent=Intent.POSITION,
                use_raw_log=True,
                reason=f"position pattern matched: {pat.pattern}",
            )

    # Topic recall: query mentions a known topic name
    if available_topics:
        for topic in available_topics:
            if topic.lower() in lower:
                return RoutingDecision(
                    intent=Intent.TOPIC_RECALL,
                    use_vector=True,
                    topic_filter=topic,
                    reason=f"references topic: {topic}",
                )

    # Semantic recall: explicit recall phrasing
    for pat in _RECALL_PATTERNS:
        if pat.search(lower):
            return RoutingDecision(
                intent=Intent.SEMANTIC_RECALL,
                use_vector=True,
                reason=f"recall pattern matched: {pat.pattern}",
            )

    # Default: normal turn — window covers it, summary handles drift
    return RoutingDecision(
        intent=Intent.NORMAL,
        reason="default — window + summary",
    )
