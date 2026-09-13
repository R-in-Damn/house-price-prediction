from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PropertyInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    bedrooms: float | None = Field(default=None, ge=0, le=50)
    bathrooms: float | None = Field(default=None, ge=0, le=50)
    area: float | None = Field(default=None, gt=0, le=1_000_000)
    zipcode: str | None = None
    asking_price: float | None = Field(default=None, gt=0)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    monthly_rent: float | None = Field(default=None, gt=0)
    holding_years: int = Field(default=5, ge=1, le=50)
    annual_appreciation_rate: float | None = Field(default=None, ge=-1, le=1)
    images: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("zipcode")
    @classmethod
    def validate_zipcode(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("zipcode cannot be blank")
        return value.strip() if value else value


class ComparableQuery(PropertyInput):
    limit: int = Field(default=10, ge=1, le=50)


class InvestmentAssumptions(BaseModel):
    purchase_cost_rate: float = Field(default=0.07, ge=0, le=1)
    annual_expense_rate: float = Field(default=0.20, ge=0, le=1)
    opportunity_rate: float = Field(default=0.06, ge=-1, le=1)


class InvestmentRequest(PropertyInput):
    assumptions: InvestmentAssumptions = Field(default_factory=InvestmentAssumptions)


class ComparableProperty(BaseModel):
    property_id: str
    bedrooms: float
    bathrooms: float
    area: float
    zipcode: str
    price: float
    price_per_area: float
    image_count: int
    distance_score: float


class ValuationResponse(BaseModel):
    estimated_value: float
    lower_bound: float
    upper_bound: float
    currency: str
    dataset: str
    method: str
    confidence: Literal["low", "medium", "high"]
    assumptions: list[str]
    image_count: int
    comparable_count: int
    price_gap: float | None = None
    price_gap_percent: float | None = None
    feature_contributions: dict[str, float] = Field(default_factory=dict)


class InvestmentResponse(BaseModel):
    fair_value: float
    asking_price: float | None
    rental_yield: float | None
    projected_value: float | None
    roi_percent: float | None
    opportunity_cost_percent: float | None
    decision: Literal["BUY", "HOLD", "SELL", "AVOID"]
    score: float
    assumptions: list[str]
    uncertainty: str
    comparable_count: int
    valuation: ValuationResponse


class ErrorResponse(BaseModel):
    detail: Any
