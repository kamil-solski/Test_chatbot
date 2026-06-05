from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from .embeddings import get_embeddings


class SessionRAGStore:
    """Per-session in-memory FAISS store for semantic retrieval of past messages."""

    def __init__(self) -> None:
        self._embeddings = get_embeddings()
        self._stores: dict[str, FAISS] = {}
        # Dedup by (role, content) tuple — msg.id is None for HumanMessage/AIMessage
        # constructed without an explicit ID, which is the common case.
        self._indexed: dict[str, set[tuple[str, str]]] = {}

    def index_messages(self, session_id: str, messages: list[BaseMessage]) -> None:
        """Embed and store any messages not yet indexed for this session."""
        indexed = self._indexed.setdefault(session_id, set())
        new_docs: list[Document] = []

        for msg in messages:
            if not isinstance(msg.content, str) or not msg.content.strip():
                continue
            role = type(msg).__name__.replace("Message", "").lower()
            key = (role, msg.content)
            if key in indexed:
                continue
            new_docs.append(Document(
                page_content=msg.content,
                metadata={"role": role, "message_id": msg.id},
            ))
            indexed.add(key)

        if not new_docs:
            return

        if session_id in self._stores:
            self._stores[session_id].add_documents(new_docs)
        else:
            self._stores[session_id] = FAISS.from_documents(new_docs, self._embeddings)

    def retrieve(self, session_id: str, query: str, k: int = 4) -> list[Document]:
        """Return top-k semantically relevant past messages for the query."""
        store = self._stores.get(session_id)
        if not store:
            return []
        return store.similarity_search(query, k=k)

    def has_session(self, session_id: str) -> bool:
        return session_id in self._stores
