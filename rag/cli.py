from __future__ import annotations

import argparse

from .service import RAGService


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest real-estate knowledge documents into the RAG vector database")
    parser.add_argument("--documents", default="rag/documents")
    args = parser.parse_args()
    service = RAGService()
    count = service.ingest_directory(__import__("pathlib").Path(args.documents))
    print(f"Ingested {count} chunks. Stats: {service.stats()}")
