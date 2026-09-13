from pathlib import Path

import pandas as pd
import pytest

from backend.app.services import RealEstateService


def make_service(tmp_path: Path) -> RealEstateService:
    archive = tmp_path / "Houses Dataset"
    archive.mkdir()
    (archive / "HousesInfo.txt").write_text(
        "3 2 1800 91901 500000\n4 3 2200 91901 650000\n2 1 1000 92021 300000\n",
        encoding="utf-8",
    )
    return RealEstateService(archive_dir=archive, model_path=tmp_path / "missing.joblib")


def test_archive_loader_and_comparables(tmp_path: Path):
    service = make_service(tmp_path)
    assert service.dataset_info()["records"] == 3
    results = service.comparables({"bedrooms": 3, "bathrooms": 2, "area": 1800, "zipcode": "91901"}, 2)
    assert len(results) == 2
    assert results[0]["property_id"] == "1"


def test_valuation_returns_uncertainty_and_gap(tmp_path: Path):
    service = make_service(tmp_path)
    result = service.valuation({"bedrooms": 3, "bathrooms": 2, "area": 1800, "zipcode": "91901", "asking_price": 450000})
    assert result["estimated_value"] > 0
    assert result["price_gap_percent"] < 0
    assert result["lower_bound"] < result["estimated_value"] < result["upper_bound"]


def test_investment_calculations(tmp_path: Path):
    service = make_service(tmp_path)
    result = service.investment(
        {"bedrooms": 3, "bathrooms": 2, "area": 1800, "zipcode": "91901", "asking_price": 500000, "monthly_rent": 3000, "annual_appreciation_rate": 0.05},
        {"opportunity_rate": 0.06},
    )
    assert result["rental_yield"] == pytest.approx(7.2)
    assert result["projected_value"] > result["fair_value"]
    assert result["decision"] in {"BUY", "HOLD", "SELL", "AVOID"}


def test_invalid_area_is_rejected():
    from backend.app.schemas import PropertyInput

    with pytest.raises(ValueError):
        PropertyInput(area=0)
