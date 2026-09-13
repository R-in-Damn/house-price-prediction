from pathlib import Path

import numpy as np

from rag.answer import generate_answer, is_legal_question
from rag.cleaning import chunk_document, clean_text
from rag.embeddings import HashEmbedding
from rag.models import Document
from rag.retriever import Retriever
from rag.service import RAGService
from rag.vector_store import SQLiteVectorStore


def test_cleaning_and_chunking_preserve_source():
    document = Document("doc-1", "Checklist", "guide.md", "  Title\n\nVerify title documents.\x00 Check permits. ")
    chunks = chunk_document(document, max_chars=200)
    assert clean_text(document.text).startswith("Title")
    assert chunks[0].source == "guide.md"
    assert "Verify title" in chunks[0].text


def test_vector_retrieval_returns_citations(tmp_path: Path):
    store = SQLiteVectorStore(tmp_path / "vectors.sqlite3")
    service = RAGService(store_path=tmp_path / "vectors.sqlite3")
    service.ingest([Document("rera", "RERA guide", "rera.md", "RERA project information depends on the applicable jurisdiction and official regulator.")])
    result = service.ask("What is RERA?", limit=3)
    assert result.citations
    assert result.citations[0]["source"] == "rera.md"
    assert result.grounded is True


def test_legal_answers_include_disclaimer():
    assert is_legal_question("What should I verify in a sale agreement?")
    result = generate_answer("What is RERA?", [], "hash-fallback")
    assert result.grounded is False
    assert result.legal_disclaimer is not None
    assert "not legal advice" in result.answer


def test_hash_embedding_is_deterministic():
    provider = HashEmbedding(dimension=32)
    first = provider.encode(["property registration"])
    second = provider.encode(["property registration"])
    assert np.array_equal(first, second)
