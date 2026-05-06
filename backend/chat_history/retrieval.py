"""Orchestrates retrieval based on a routing decision.

Maps each Intent to one of the four retrieval patterns:
  - CONTINUATION → sliding window only
  - POSITION     → filter-only on raw log (no embeddings)
  - TOPIC_RECALL → filter→vector (topic-bounded similarity)
  - SEMANTIC_RECALL → pure vector
  - NORMAL       → window only (summary handles drift via separate node)
"""
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from .router import Intent, RoutingDecision
from .store import ChatHistoryStore


async def retrieve_for_query(
    decision: RoutingDecision,
    messages: list[BaseMessage],
    store: ChatHistoryStore,
    session_id: str,
    query: str,
    rag_k: int = 4,
    window_size: int = 6,
) -> tuple[list[BaseMessage], str]:
    """Returns (messages_for_llm, extra_system_context)."""
    current = messages[-1]
    history = messages[:-1]

    # Always index messages so future turns can retrieve them
    await store.index_messages(session_id, history)

    if decision.intent == Intent.POSITION:
        position_info = _resolve_position(query, history)
        return [current], f"Relevant raw log entry:\n{position_info}"

    if decision.use_vector:
        docs = store.retrieve(
            session_id, query, k=rag_k, topic=decision.topic_filter
        )
        role_map = {"human": HumanMessage, "ai": AIMessage}
        retrieved = [
            role_map.get(d.metadata.get("role"), HumanMessage)(content=d.page_content)
            for d in docs
        ]
        return retrieved + [current], ""

    # CONTINUATION / NORMAL: sliding window only
    return history[-window_size:] + [current], ""


def _resolve_position(query: str, history: list[BaseMessage]) -> str:
    lower = query.lower()
    user_messages = [m for m in history if isinstance(m, HumanMessage)]

    if not user_messages:
        return "(no prior user prompts in this session)"

    if any(w in lower for w in ("first", "initial", "original", "starting")):
        return f"User's first prompt: {user_messages[0].content}"

    if any(w in lower for w in ("last", "previous", "prior")):
        return f"User's most recent prior prompt: {user_messages[-1].content}"

    match = re.search(r"\b(\d+)(st|nd|rd|th)?\s+(prompt|message|question)\b", lower)
    if match:
        n = int(match.group(1))
        if 1 <= n <= len(user_messages):
            return f"User's prompt #{n}: {user_messages[n - 1].content}"
        return f"(prompt #{n} not found — only {len(user_messages)} prompts so far)"

    return f"(could not resolve position; total prior prompts: {len(user_messages)})"
