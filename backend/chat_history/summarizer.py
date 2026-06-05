"""Summarizes batches of messages that have left the sliding window."""
from typing import Awaitable, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage

OnSummaryCreated = Callable[[str, int, list[BaseMessage], str], Awaitable[None]]


class ChunkSummarizer:
    def __init__(
        self,
        llm: BaseChatModel,
        chunk_size: int = 6,
        on_summary_created: OnSummaryCreated | None = None,
    ) -> None:
        self._llm = llm
        self._chunk_size = chunk_size
        self._summarized: dict[str, int] = {}    # session_id -> messages summarized so far
        self._chunks: dict[str, list[str]] = {}  # session_id -> ordered chunk summaries
        self._on_summary_created = on_summary_created

    async def update(self, session_id: str, history: list[BaseMessage], window_size: int) -> None:
        """Summarize any complete chunks of messages that have fallen outside the window."""
        if len(history) <= window_size:
            return

        outside = history[:-window_size]
        already = self._summarized.get(session_id, 0)
        pending = outside[already:]

        while len(pending) >= self._chunk_size:
            batch = pending[: self._chunk_size]
            summary = await self._summarize_batch(batch)
            chunks = self._chunks.setdefault(session_id, [])
            chunks.append(summary)
            chunk_index = len(chunks) - 1
            already += self._chunk_size
            self._summarized[session_id] = already
            pending = pending[self._chunk_size :]

            if self._on_summary_created is not None:
                await self._on_summary_created(session_id, chunk_index, batch, summary)

    async def _summarize_batch(self, messages: list[BaseMessage]) -> str:
        lines = []
        for m in messages:
            if not isinstance(m.content, str):
                continue
            role = "User" if isinstance(m, HumanMessage) else "Assistant"
            lines.append(f"{role}: {m.content}")
        prompt = "Summarize this conversation excerpt concisely in 2-4 sentences:\n\n" + "\n".join(lines)
        response = await self._llm.ainvoke([HumanMessage(content=prompt)])
        return response.content

    def get_context(self, session_id: str) -> str:
        chunks = self._chunks.get(session_id, [])
        if not chunks:
            return ""
        parts = [f"[Summary {i + 1}]\n{s}" for i, s in enumerate(chunks)]
        return "Earlier conversation summaries:\n\n" + "\n\n".join(parts)
