from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .embeddings import get_embeddings

_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


class DocumentRAGStore:
    """Per-session FAISS store for user-uploaded document chunks."""

    def __init__(self) -> None:
        self._embeddings = get_embeddings()
        self._stores: dict[str, FAISS] = {}
        self._docs: dict[str, list[str]] = {}   # session_id -> list of filenames

    def add_document(self, session_id: str, filename: str, text: str) -> int:
        chunks = _SPLITTER.split_text(text)
        if not chunks:
            return 0

        docs = [
            Document(page_content=c, metadata={"source": filename, "chunk": i})
            for i, c in enumerate(chunks)
        ]

        if session_id in self._stores:
            self._stores[session_id].add_documents(docs)
        else:
            self._stores[session_id] = FAISS.from_documents(docs, self._embeddings)

        self._docs.setdefault(session_id, [])
        if filename not in self._docs[session_id]:
            self._docs[session_id].append(filename)

        return len(chunks)

    def retrieve(self, session_id: str, query: str, k: int = 4) -> list[Document]:
        store = self._stores.get(session_id)
        if not store:
            return []
        return store.similarity_search(query, k=k)

    def list_documents(self, session_id: str) -> list[str]:
        return self._docs.get(session_id, [])

    def clear(self, session_id: str) -> None:
        self._stores.pop(session_id, None)
        self._docs.pop(session_id, None)

    def has_documents(self, session_id: str) -> bool:
        return session_id in self._stores
