# Estate Signal

Production-style real-estate intelligence workspace built on the existing house-price ML pipeline.

## What is included

- Preserved Ames scikit-learn pipeline and training script.
- Dataset adapter for `data/archive (1)/Houses Dataset`.
- Archive metadata and property-image discovery.
- Optional pretrained ResNet18 image embeddings with structured plus image late fusion.
- FastAPI REST API for valuation, comparables, and investment analysis.
- React + TypeScript frontend replacing Streamlit as the primary UI.
- Transparent valuation ranges, assumptions, confidence, and decision-support scores.
- Pytest coverage for services, validation, comparables, investment calculations, and API endpoints.
- Docker Compose definitions for backend, frontend, and PostgreSQL.
- Separate document-grounded RAG assistant with citations and legal-safety guardrails.

The archive contains 535 US properties with USD prices, ZIP codes, and four images per property. Rental and historical market calculations are shown as unavailable unless the required inputs are supplied. Results are decision support, not financial advice.

## Local development

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn backend.app.main:app --reload
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. API documentation is available at `http://localhost:8000/docs`.

## RAG knowledge assistant

Ingest the included general reference document before asking questions:

```bash
python -m rag.cli --documents rag/documents
```

The RAG API is available at `POST /api/v1/rag/query`, `POST /api/v1/rag/ingest`, and `GET /api/v1/rag/stats`. Set `OPENAI_API_KEY` to enable answer generation with retrieved context; without it, the system uses a conservative extractive fallback. Set `RAG_USE_HASH_EMBEDDINGS=true` for deterministic offline development, otherwise `sentence-transformers` uses `all-MiniLM-L6-v2`.

Legal and transaction answers are general information only, include retrieved source citations, and state that requirements may depend on state or local jurisdiction. The assistant refuses to invent laws, thresholds, deadlines, or procedures absent from the indexed sources.

## Existing ML pipeline

The original pipeline remains available:

```bash
python -m src.train_model --data_dir data --target SalePrice
```

The API uses the preserved Ames model when its original feature schema is supplied. Otherwise it uses the archive's comparable-property estimate.

## Train the multimodal model

Install the vision dependencies from `requirements.txt`, then run:

```bash
python -m src.train_multimodal
```

This downloads the pretrained ResNet18 weights on first use, averages embeddings across each property's available images, fuses them with bedrooms, bathrooms, area, and ZIP code, and saves `artifacts/multimodal_fusion.joblib`. The API automatically uses this artifact when image data is supplied.

## Docker

```bash
docker compose up --build
```

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`
- PostgreSQL: `localhost:5432`

PostgreSQL is provisioned for the persistence phase; the current MVP reads the archive directly so the existing data remains immediately usable.
