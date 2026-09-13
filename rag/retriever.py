from __future__ import annotations

from .cleaning import keywords
from .embeddings import get_embedding_provider
from .models import RetrievedChunk
from .vector_store import SQLiteVectorStore


class Retriever:
    def __init__(self, store: SQLiteVectorStore, embedding_provider=None) -> None:
        self.store = store
        self.embedding_provider = embedding_provider or get_embedding_provider()

    def search(self, question: str, limit: int = 5) -> list[RetrievedChunk]:
        if not question.strip():
            raise ValueError("question cannot be blank")
        query_embedding = self.embedding_provider.encode([question])[0]
        results = self.store.search(query_embedding, limit=limit)
        query_terms = keywords(question)
        # Add a small lexical tie-breaker so exact legal terms survive weak fallback embeddings.
        return sorted(results, key=lambda item: (len(query_terms & keywords(item.text)) * 0.01 + item.score), reverse=True)
