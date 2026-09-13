from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from rag.models import Document
from rag.service import rag_service

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])


class RAGQuery(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    limit: int = Field(default=5, ge=1, le=10)


class RAGDocument(BaseModel):
    document_id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=300)
    source: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=20, max_length=500_000)
    metadata: dict[str, Any] = Field(default_factory=dict)


@router.get("/stats")
def rag_stats() -> dict[str, object]:
    return rag_service.stats()


@router.post("/ingest")
def rag_ingest(documents: list[RAGDocument]) -> dict[str, object]:
    if not documents:
        raise HTTPException(status_code=400, detail="At least one document is required")
    count = rag_service.ingest([Document(**document.model_dump()) for document in documents])
    return {"chunks_ingested": count, **rag_service.stats()}


@router.post("/query")
def rag_query(payload: RAGQuery) -> dict[str, object]:
    try:
        result = rag_service.ask(payload.question, payload.limit)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {
        "answer": result.answer,
        "citations": result.citations,
        "grounded": result.grounded,
        "legal_disclaimer": result.legal_disclaimer,
        "retrieval_count": result.retrieval_count,
        "embedding_provider": result.embedding_provider,
        "llm_provider": result.llm_provider,
    }
