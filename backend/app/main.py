from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .schemas import ComparableQuery, InvestmentRequest, PropertyInput
from .services import service
from .rag_routes import router as rag_router

app = FastAPI(title="Real Estate Intelligence API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(rag_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/model-info")
def model_info() -> dict[str, object]:
    return service.dataset_info()


@app.post("/api/v1/valuation")
def valuation(payload: PropertyInput) -> dict[str, object]:
    return service.valuation(payload.model_dump())


@app.post("/api/v1/comparables/search")
def comparables(payload: ComparableQuery) -> list[dict[str, object]]:
    return service.comparables(payload.model_dump(), payload.limit)


@app.post("/api/v1/investment-analysis")
def investment_analysis(payload: InvestmentRequest) -> dict[str, object]:
    data = payload.model_dump()
    assumptions = data.pop("assumptions")
    return service.investment(data, assumptions)
