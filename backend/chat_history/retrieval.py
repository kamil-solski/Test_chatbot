"""Orchestrates retrieval based on a routing decision.

Maps each Intent to one of the four retrieval patterns:
  - CONTINUATION → sliding window + chunk summaries
  - POSITION     → filter-only on raw log (no embeddings) + chunk summaries
  - TOPIC_RECALL → filter→vector (topic-bounded similarity) + chunk summaries
  - SEMANTIC_RECALL → pure vector + chunk summaries
  - NORMAL       → sliding window + chunk summaries
"""
import re
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from .router import Intent, RoutingDecision
from .store import ChatHistoryStore

if TYPE_CHECKING:
    from .summarizer import ChunkSummarizer


async def retrieve_for_query(
    decision: RoutingDecision,
    messages: list[BaseMessage],
    store: ChatHistoryStore,
    session_id: str,
    query: str,
    rag_k: int = 4,
    window_size: int = 6,
    summarizer: "ChunkSummarizer | None" = None,
) -> tuple[list[BaseMessage], str, list[dict]]:
    """Returns (messages_for_llm, extra_system_context, retrieved_metadata).

    retrieved_metadata is a list of {role, content} dicts representing what
    came back from vector retrieval (empty for non-vector paths).
    """
    current = messages[-1]
    history = messages[:-1]

    await store.index_messages(session_id, history)

    if summarizer:
        await summarizer.update(session_id, history, window_size)
    chunk_context = summarizer.get_context(session_id) if summarizer else ""

    if decision.intent == Intent.POSITION:
        position_info = _resolve_position(query, history)
        extra = f"Relevant raw log entry:\n{position_info}"
        if chunk_context:
            extra = chunk_context + "\n\n" + extra
        return [current], extra, []

    if decision.use_vector:
        docs = store.retrieve(
            session_id, query, k=rag_k, topic=decision.topic_filter
        )
        role_map = {"human": HumanMessage, "ai": AIMessage}
        retrieved = [
            role_map.get(d.metadata.get("role"), HumanMessage)(content=d.page_content)
            for d in docs
        ]
        retrieved_meta = [
            {"role": d.metadata.get("role", "human"), "content": d.page_content}
            for d in docs
        ]
        return retrieved + [current], chunk_context, retrieved_meta

    # CONTINUATION / NORMAL: sliding window + chunk summaries
    return history[-window_size:] + [current], chunk_context, []


def _resolve_position(query: str, history: list[BaseMessage]) -> str:
    lower = query.lower()
    user_messages = [m for m in history if isinstance(m, HumanMessage)]

    if not user_messages:
        return "(no prior user prompts in this session)"

    topic_query = any(w in lower for w in ("topic", "subject", "thing", "discussion"))

    if any(w in lower for w in ("first", "initial", "original", "starting")):
        if topic_query:
            # Return first two exchanges so the model can infer the actual topic
            excerpt = _format_exchanges(history[:4])
            return f"Start of conversation:\n{excerpt}"
        return f"User's first prompt: {user_messages[0].content}"

    if any(w in lower for w in ("last", "previous", "prior")):
        if topic_query:
            excerpt = _format_exchanges(history[-4:])
            return f"Most recent conversation:\n{excerpt}"
        return f"User's most recent prior prompt: {user_messages[-1].content}"

    match = re.search(r"\b(\d+)(st|nd|rd|th)?\s+(prompt|message|question)\b", lower)
    if match:
        n = int(match.group(1))
        if 1 <= n <= len(user_messages):
            return f"User's prompt #{n}: {user_messages[n - 1].content}"
        return f"(prompt #{n} not found — only {len(user_messages)} prompts so far)"

    return f"(could not resolve position; total prior prompts: {len(user_messages)})"


def _format_exchanges(messages: list[BaseMessage]) -> str:
    lines = []
    for m in messages:
        if not isinstance(m.content, str):
            continue
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        lines.append(f"{role}: {m.content[:300]}")
    return "\n".join(lines)
