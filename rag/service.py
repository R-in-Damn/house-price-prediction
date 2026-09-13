from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from .answer import generate_answer
from .cleaning import chunk_document
from .models import Document
from .retriever import Retriever
from .vector_store import SQLiteVectorStore


ROOT = Path(__file__).resolve().parent
DEFAULT_STORE = ROOT / "storage" / "rag.sqlite3"
DEFAULT_DOCUMENTS = ROOT / "documents"


class RAGService:
    def __init__(self, store_path: Path = DEFAULT_STORE) -> None:
        self.store = SQLiteVectorStore(store_path)
        self.retriever = Retriever(self.store)

    def ingest(self, documents: Iterable[Document]) -> int:
        documents = list(documents)
        chunks = [chunk for document in documents for chunk in chunk_document(document)]
        if not chunks:
            return 0
        embeddings = self.retriever.embedding_provider.encode([chunk.text for chunk in chunks])
        self.store.upsert(chunks, embeddings)
        return len(chunks)

    def ingest_directory(self, directory: Path = DEFAULT_DOCUMENTS) -> int:
        documents = []
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".pdf"}:
                continue
            if path.suffix.lower() == ".pdf":
                try:
                    from pypdf import PdfReader
                    text = "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
                except ImportError as error:
                    raise RuntimeError("Install pypdf to ingest PDF documents") from error
            else:
                text = path.read_text(encoding="utf-8")
            document_id = hashlib.sha256(str(path.relative_to(directory)).encode()).hexdigest()[:16]
            documents.append(Document(document_id, path.stem.replace("_", " "), str(path.relative_to(directory)), text, {"file_type": path.suffix.lower()}))
        return self.ingest(documents)

    def ask(self, question: str, limit: int = 5):
        chunks = self.retriever.search(question, limit=limit)
        return generate_answer(question, chunks, self.retriever.embedding_provider.name)

    def stats(self) -> dict[str, object]:
        return {"chunks": self.store.count(), "embedding_provider": self.retriever.embedding_provider.name, "store": str(self.store.path)}


rag_service = RAGService()
