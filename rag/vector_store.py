from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from .models import Chunk, RetrievedChunk


class SQLiteVectorStore:
    """Persistent local vector database with cosine similarity retrieval."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    source TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    metadata TEXT NOT NULL
                )
            """)
            connection.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have one embedding")
        with self._connect() as connection:
            connection.executemany(
                """INSERT OR REPLACE INTO chunks
                (chunk_id, document_id, title, source, text, embedding, dimension, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [(
                    chunk.chunk_id, chunk.document_id, chunk.title, chunk.source, chunk.text,
                    np.asarray(embedding, dtype=np.float32).tobytes(), len(embedding), json.dumps(chunk.metadata),
                ) for chunk, embedding in zip(chunks, embeddings)],
            )

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def search(self, embedding: np.ndarray, limit: int = 5, minimum_score: float = 0.08) -> list[RetrievedChunk]:
        query = np.asarray(embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if not query_norm:
            return []
        results: list[RetrievedChunk] = []
        with self._connect() as connection:
            rows = connection.execute("SELECT chunk_id, document_id, title, source, text, embedding, dimension, metadata FROM chunks").fetchall()
        for chunk_id, document_id, title, source, text, raw_embedding, dimension, metadata in rows:
            vector = np.frombuffer(raw_embedding, dtype=np.float32, count=dimension)
            denominator = query_norm * np.linalg.norm(vector)
            score = float(np.dot(query, vector) / denominator) if denominator else 0.0
            if score >= minimum_score:
                results.append(RetrievedChunk(chunk_id, document_id, title, source, text, score, json.loads(metadata)))
        return sorted(results, key=lambda item: item.score, reverse=True)[:limit]
