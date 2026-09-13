from __future__ import annotations

import re

from .models import Chunk, Document

_WORD_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9'-]{2,}\b")


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _sentences(text: str) -> list[str]:
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    sentences: list[str] = []
    for paragraph in paragraphs:
        sentences.extend(part.strip() for part in re.split(r"(?<=[.!?])\s+", paragraph) if part.strip())
    return sentences


def chunk_document(document: Document, max_chars: int = 1200, overlap_chars: int = 180) -> list[Chunk]:
    if max_chars < 200 or overlap_chars >= max_chars:
        raise ValueError("max_chars must be at least 200 and greater than overlap_chars")
    text = clean_text(document.text)
    if not text:
        return []
    chunks: list[Chunk] = []
    current = ""
    for sentence in _sentences(text):
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunk_number = len(chunks)
            chunks.append(Chunk(f"{document.document_id}:{chunk_number}", document.document_id, document.title, document.source, current, document.metadata))
            current = current[-overlap_chars:] + " " + sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(Chunk(f"{document.document_id}:{len(chunks)}", document.document_id, document.title, document.source, current, document.metadata))
    return chunks


def keywords(text: str) -> set[str]:
    return {match.group(0).lower() for match in _WORD_RE.finditer(text)}
