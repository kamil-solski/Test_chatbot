"""Per-session FAISS store for chat messages, with topic metadata for filtering."""
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from rag.embeddings import get_embeddings

from .clustering import TopicClusterer


class ChatHistoryStore:
    def __init__(self, clusterer: TopicClusterer) -> None:
        self._embeddings = get_embeddings()
        self._stores: dict[str, FAISS] = {}
        self._indexed: dict[str, set] = {}
        self._clusterer = clusterer

    async def index_messages(self, session_id: str, messages: list[BaseMessage]) -> None:
        """Embed and store any messages not yet indexed for this session.
        Each message is also assigned a topic via the clusterer."""
        indexed = self._indexed.setdefault(session_id, set())
        new_docs: list[Document] = []

        for msg in messages:
            msg_id = msg.id
            if msg_id in indexed or not isinstance(msg.content, str) or not msg.content.strip():
                continue
            role = type(msg).__name__.replace("Message", "").lower()
            topic = await self._clusterer.assign_topic(session_id, msg_id, msg.content)
            new_docs.append(Document(
                page_content=msg.content,
                metadata={"role": role, "message_id": msg_id, "topic": topic},
            ))
            indexed.add(msg_id)

        if not new_docs:
            return

        if session_id in self._stores:
            self._stores[session_id].add_documents(new_docs)
        else:
            self._stores[session_id] = FAISS.from_documents(new_docs, self._embeddings)

    def retrieve(
        self,
        session_id: str,
        query: str,
        k: int = 4,
        topic: str | None = None,
    ) -> list[Document]:
        store = self._stores.get(session_id)
        if not store:
            return []
        if topic:
            return store.similarity_search(
                query,
                k=k,
                filter=lambda meta: meta.get("topic") == topic,
            )
        return store.similarity_search(query, k=k)

    def list_topics(self, session_id: str) -> list[str]:
        return self._clusterer.list_topics(session_id)

    def has_session(self, session_id: str) -> bool:
        return session_id in self._stores
