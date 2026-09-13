from __future__ import annotations

import hashlib
import os
import re
from typing import Iterable

import numpy as np


class HashEmbedding:
    """Deterministic offline fallback; use a semantic provider in production."""

    name = "hash-fallback"

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            vector = np.zeros(self.dimension, dtype=np.float32)
            tokens = re.findall(r"[a-z0-9]+", text.lower())
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimension
                vector[index] += 1.0 if digest[4] % 2 else -1.0
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm else vector)
        return np.vstack(vectors) if vectors else np.empty((0, self.dimension), dtype=np.float32)


class SentenceTransformerEmbedding:
    name = "sentence-transformers"

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name or os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True), dtype=np.float32)


def get_embedding_provider():
    if os.getenv("RAG_USE_HASH_EMBEDDINGS", "false").lower() == "true":
        return HashEmbedding()
    try:
        return SentenceTransformerEmbedding()
    except ImportError:
        return HashEmbedding()
