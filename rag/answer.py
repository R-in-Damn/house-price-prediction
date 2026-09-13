from __future__ import annotations

import os
import re
from typing import Any

from .cleaning import keywords
from .models import Answer, RetrievedChunk

LEGAL_TERMS = {"legal", "law", "rera", "agreement", "registration", "title", "deed", "stamp", "due", "diligence", "verify", "jurisdiction", "document"}


def is_legal_question(question: str) -> bool:
    return bool(keywords(question) & LEGAL_TERMS)


def _citations(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    return [{"id": f"S{index}", "source": chunk.source, "title": chunk.title, "chunk_id": chunk.chunk_id, "score": round(chunk.score, 4)} for index, chunk in enumerate(chunks, 1)]


def _extractive_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    question_terms = keywords(question)
    candidates: list[tuple[int, str]] = []
    for chunk in chunks:
        for sentence in re.split(r"(?<=[.!?])\s+", chunk.text):
            overlap = len(question_terms & keywords(sentence))
            if overlap:
                candidates.append((overlap, sentence.strip()))
    selected = [sentence for _, sentence in sorted(candidates, reverse=True)[:4]]
    if not selected:
        return "I could not find enough relevant information in the indexed source documents to answer this reliably."
    return " ".join(dict.fromkeys(selected))


def _openai_answer(question: str, chunks: list[RetrievedChunk]) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI

        context = "\n\n".join(f"[S{index}] {chunk.title} ({chunk.source})\n{chunk.text}" for index, chunk in enumerate(chunks, 1))
        prompt = (
            "Answer only from the supplied sources. Do not invent statutes, deadlines, forms, authorities, or procedures. "
            "If the sources do not support an answer, say so. Cite claims inline as [S1], [S2]. "
            "For legal or property transaction questions, distinguish general information from legal advice and state that requirements depend on the applicable state/local jurisdiction.\n\n"
            f"Question: {question}\n\nSources:\n{context}"
        )
        response = OpenAI(api_key=api_key).chat.completions.create(
            model=os.getenv("RAG_LLM_MODEL", "gpt-4o-mini"),
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a cautious, source-grounded real-estate information assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception:
        return None


def generate_answer(question: str, chunks: list[RetrievedChunk], embedding_provider: str) -> Answer:
    legal = is_legal_question(question)
    citations = _citations(chunks)
    if not chunks:
        text = "I could not find a relevant indexed source document. Please add authoritative documents for this question before relying on an answer."
        grounded = False
        provider = "none"
    else:
        text = _openai_answer(question, chunks)
        provider = "openai" if text else "extractive-fallback"
        if text is None:
            text = _extractive_answer(question, chunks)
        grounded = True
    disclaimer = None
    if legal:
        disclaimer = "General information only, not legal advice. Property, RERA, registration, agreement, and due-diligence requirements can vary by state, local authority, property type, and transaction facts. Consult a qualified local lawyer or the relevant authority."
        text = f"{text}\n\n{disclaimer}"
    return Answer(text, citations, grounded, disclaimer, len(chunks), embedding_provider, provider)
