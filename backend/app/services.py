from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from joblib import load
except ImportError:  # Optional until the ML dependencies are installed.
    load = None

try:
    from .multimodal import ImageEncoder, LateFusionRegressor, decode_data_url
except ImportError:
    ImageEncoder = None
    LateFusionRegressor = None
    decode_data_url = None


ARCHIVE_DIR = Path(__file__).resolve().parents[2] / "data" / "archive (1)" / "Houses Dataset"
MODEL_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "model_pipeline.joblib"
MULTIMODAL_MODEL_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "multimodal_fusion.joblib"


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    currency: str
    country: str
    records: int
    images: int
    has_rent: bool
    has_history: bool
    has_image_model: bool


class RealEstateService:
    def __init__(self, archive_dir: Path = ARCHIVE_DIR, model_path: Path = MODEL_PATH) -> None:
        self.archive_dir = archive_dir
        self.records = self._load_archive(archive_dir)
        self.model = load(model_path) if load is not None and model_path.exists() else None
        self.fusion_model = load(MULTIMODAL_MODEL_PATH) if load is not None and MULTIMODAL_MODEL_PATH.exists() else None
        self.image_encoder = None
        self.info = DatasetInfo(
            name="Houses Dataset",
            currency="USD",
            country="United States",
            records=len(self.records),
            images=self._image_count(archive_dir),
            has_rent=False,
            has_history=False,
            has_image_model=self.fusion_model is not None,
        )

    @staticmethod
    def _load_archive(directory: Path) -> pd.DataFrame:
        metadata = directory / "HousesInfo.txt"
        if not metadata.exists():
            return pd.DataFrame(columns=["bedrooms", "bathrooms", "area", "zipcode", "price"])
        rows = []
        for line in metadata.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            bedrooms, bathrooms, area, zipcode, price = parts
            rows.append({
                "property_id": str(len(rows) + 1),
                "bedrooms": float(bedrooms),
                "bathrooms": float(bathrooms),
                "area": float(area),
                "zipcode": zipcode,
                "price": float(price),
            })
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame["price_per_area"] = frame["price"] / frame["area"]
        return frame

    @staticmethod
    def _image_count(directory: Path) -> int:
        return len(list(directory.glob("*.jpg"))) if directory.exists() else 0

    def dataset_info(self) -> dict[str, Any]:
        return self.info.__dict__.copy()

    def _archive_comparables(self, prop: dict[str, Any], limit: int = 10) -> list[dict[str, Any]]:
        if self.records.empty:
            return []
        frame = self.records.copy()
        for field in ("bedrooms", "bathrooms", "area"):
            value = prop.get(field)
            if value is not None:
                frame[f"{field}_distance"] = (frame[field] - float(value)).abs() / max(float(value), 1.0)
            else:
                frame[f"{field}_distance"] = 0.5
        zipcode = str(prop.get("zipcode")) if prop.get("zipcode") is not None else None
        frame["zip_distance"] = (frame["zipcode"] != zipcode).astype(float) if zipcode else 0.25
        frame["distance_score"] = (
            frame["area_distance"] * 0.45
            + frame["bedrooms_distance"] * 0.25
            + frame["bathrooms_distance"] * 0.20
            + frame["zip_distance"] * 0.10
        )
        frame = frame.sort_values("distance_score").head(limit)
        result = []
        for _, row in frame.iterrows():
            image_count = len(list(self.archive_dir.glob(f"{row.property_id}_*.jpg")))
            result.append({
                "property_id": str(row.property_id),
                "bedrooms": float(row.bedrooms),
                "bathrooms": float(row.bathrooms),
                "area": float(row.area),
                "zipcode": str(row.zipcode),
                "price": float(row.price),
                "price_per_area": float(row.price_per_area),
                "image_count": image_count,
                "distance_score": float(row.distance_score),
            })
        return result

    def comparables(self, prop: dict[str, Any], limit: int = 10) -> list[dict[str, Any]]:
        return self._archive_comparables(prop, limit)

    def _model_prediction(self, prop: dict[str, Any]) -> float | None:
        required = {"LotArea", "GrLivArea", "TotalBsmtSF", "1stFlrSF", "GarageCars", "FullBath", "YearBuilt", "Neighborhood", "HouseStyle", "KitchenQual", "ExterQual"}
        if self.model is None or not required.issubset(prop):
            return None
        frame = pd.DataFrame([{key: prop[key] for key in required}])
        return float(self.model.predict(frame)[0])

    def _resolve_image_paths(self, values: list[str]) -> list[Path]:
        paths = []
        for value in values:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = self.archive_dir / candidate.name
            candidate = candidate.resolve()
            if candidate.is_file() and self.archive_dir.resolve() in candidate.parents:
                paths.append(candidate)
        return paths

    def _multimodal_prediction(self, prop: dict[str, Any]) -> tuple[float, dict[str, float], int] | None:
        if self.fusion_model is None or ImageEncoder is None:
            return None
        values = prop.get("images", [])
        if not values:
            return None
        if self.image_encoder is None:
            self.image_encoder = ImageEncoder()
        data_urls = [value for value in values if value.startswith("data:")]
        paths = self._resolve_image_paths([value for value in values if not value.startswith("data:")])
        if data_urls:
            embedding = self.image_encoder.encode_bytes([decode_data_url(value) for value in data_urls])
            image_count = len(data_urls) + len(paths)
            if paths:
                path_embedding = self.image_encoder.encode_paths(paths)
                embedding = (embedding + path_embedding) / 2
        elif paths:
            embedding = self.image_encoder.encode_paths(paths)
            image_count = len(paths)
        else:
            return None
        prediction = self.fusion_model.predict(prop, embedding)
        return prediction, self.fusion_model.explain(prop, embedding), image_count

    def valuation(self, prop: dict[str, Any]) -> dict[str, Any]:
        multimodal = self._multimodal_prediction(prop)
        model_value = self._model_prediction(prop)
        comps = self.comparables(prop)
        assumptions = ["This is decision support, not guaranteed financial advice."]
        image_count = multimodal[2] if multimodal is not None else len(prop.get("images", []))
        if multimodal is not None:
            estimated = multimodal[0]
            method = "Late-fusion structured plus ResNet18 image model"
            confidence = "medium"
            contributions = multimodal[1]
            assumptions.append("A pretrained ResNet18 image embedding is averaged across supplied property images and fused with structured features.")
        elif model_value is not None:
            estimated = model_value
            method = "Existing Ames structured ML pipeline"
            confidence = "medium"
            contributions = {}
            assumptions.append("Prediction uses the preserved Ames model pipeline and its original feature schema.")
        elif comps:
            weights = [1 / (1 + item["distance_score"]) for item in comps]
            estimated = sum(item["price"] * weight for item, weight in zip(comps, weights)) / sum(weights)
            method = "Comparable-property weighted estimate"
            confidence = "medium" if len(comps) >= 5 else "low"
            contributions = {}
            assumptions.append("Estimate is based on available archive comparables; no current-market adjustment is applied.")
        else:
            estimated = 0.0
            method = "Unavailable"
            confidence = "low"
            assumptions.append("Insufficient comparable or structured training data.")
            contributions = {}
        spread = max(estimated * (0.10 if confidence == "medium" else 0.20), 1.0)
        asking = prop.get("asking_price")
        gap = float(asking - estimated) if asking is not None and estimated else None
        return {
            "estimated_value": round(estimated, 2),
            "lower_bound": round(max(0, estimated - spread), 2),
            "upper_bound": round(estimated + spread, 2),
            "currency": self.info.currency,
            "dataset": self.info.name,
            "method": method,
            "confidence": confidence,
            "assumptions": assumptions,
            "image_count": image_count,
            "comparable_count": len(comps),
            "price_gap": round(gap, 2) if gap is not None else None,
            "price_gap_percent": round(gap / estimated * 100, 2) if gap is not None and estimated else None,
            "feature_contributions": contributions,
        }

    def investment(self, prop: dict[str, Any], assumptions: dict[str, Any]) -> dict[str, Any]:
        valuation = self.valuation(prop)
        fair = valuation["estimated_value"]
        asking = prop.get("asking_price")
        monthly_rent = prop.get("monthly_rent")
        rental_yield = (monthly_rent * 12 / asking * 100) if monthly_rent and asking else None
        growth = prop.get("annual_appreciation_rate")
        projected = fair * ((1 + growth) ** prop.get("holding_years", 5)) if growth is not None and fair else None
        roi = ((projected - asking) / asking * 100) if projected and asking else None
        opportunity = ((1 + assumptions.get("opportunity_rate", 0.06)) ** prop.get("holding_years", 5) - 1) * 100
        score = 50.0
        if valuation["price_gap_percent"] is not None:
            score += max(-30, min(30, -valuation["price_gap_percent"] * 0.5))
        if rental_yield is not None:
            score += max(-10, min(15, rental_yield - 4))
        if roi is not None:
            score += max(-15, min(15, roi - opportunity))
        score = round(max(0, min(100, score)), 2)
        decision = "BUY" if score >= 65 else "HOLD" if score >= 50 else "SELL" if score >= 35 else "AVOID"
        assumptions_list = valuation["assumptions"] + [
            f"Opportunity benchmark is {assumptions.get('opportunity_rate', 0.06) * 100:.1f}% annually.",
            "Rental yield and ROI are omitted when the required inputs are unavailable.",
        ]
        return {
            "fair_value": fair,
            "asking_price": asking,
            "rental_yield": round(rental_yield, 2) if rental_yield is not None else None,
            "projected_value": round(projected, 2) if projected is not None else None,
            "roi_percent": round(roi, 2) if roi is not None else None,
            "opportunity_cost_percent": round(opportunity, 2),
            "decision": decision,
            "score": score,
            "assumptions": assumptions_list,
            "uncertainty": f"{valuation['confidence']} confidence; estimated range ${valuation['lower_bound']:,.0f}-${valuation['upper_bound']:,.0f}.",
            "comparable_count": valuation["comparable_count"],
            "valuation": valuation,
        }


service = RealEstateService()
