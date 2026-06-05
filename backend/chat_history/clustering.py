"""Embedding-based clustering with LLM-named topics.

Each new message is embedded and compared against existing cluster centroids.
If similarity to the nearest cluster exceeds a threshold, the message joins that
cluster. Otherwise a new cluster is created and a cheap LLM call names it.
"""
import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI


class TopicClusterer:
    def __init__(
        self,
        embeddings: Embeddings,
        llm: ChatOpenAI,
        similarity_threshold: float = 0.65,
    ) -> None:
        self._embeddings = embeddings
        self._llm = llm
        self._threshold = similarity_threshold
        # session_id -> {"clusters": [{"centroid": np.ndarray, "name": str, "count": int}],
        #                "seen_texts": {text: topic_name}}
        # Cached by text content because msg.id is None for messages constructed
        # without an explicit id (the common case).
        self._sessions: dict[str, dict] = {}

    async def assign_topic(self, session_id: str, text: str) -> str:
        sess = self._sessions.setdefault(
            session_id, {"clusters": [], "seen_texts": {}}
        )

        if text in sess["seen_texts"]:
            return sess["seen_texts"][text]

        embedding = np.array(await self._embeddings.aembed_query(text), dtype=np.float32)

        best_idx, best_sim = -1, -1.0
        for i, cluster in enumerate(sess["clusters"]):
            sim = _cosine(embedding, cluster["centroid"])
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_idx >= 0 and best_sim >= self._threshold:
            cluster = sess["clusters"][best_idx]
            n = cluster["count"]
            cluster["centroid"] = (cluster["centroid"] * n + embedding) / (n + 1)
            cluster["count"] = n + 1
            topic = cluster["name"]
        else:
            topic = await self._name_topic(text)
            sess["clusters"].append({
                "centroid": embedding,
                "name": topic,
                "count": 1,
            })

        sess["seen_texts"][text] = topic
        return topic

    async def _name_topic(self, text: str) -> str:
        prompt = (
            "Suggest a short topic label (2-4 words) for this message. "
            "Reply with only the label — no quotes, no punctuation, no explanation.\n\n"
            f"Message: {text[:500]}"
        )
        response = await self._llm.ainvoke(prompt)
        label = response.content.strip().strip('"\'.,')
        # Normalize to title case for stable filtering
        return label or "General"

    def list_topics(self, session_id: str) -> list[str]:
        sess = self._sessions.get(session_id)
        if not sess:
            return []
        return [c["name"] for c in sess["clusters"]]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(a @ b / denom)
