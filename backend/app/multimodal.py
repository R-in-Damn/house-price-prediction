from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ImportError:
    Ridge = None
    OneHotEncoder = None
    StandardScaler = None

try:
    import torch
    from torchvision.models import ResNet18_Weights, resnet18
except ImportError:  # Vision is optional for structured-only deployments.
    torch = None
    ResNet18_Weights = None
    resnet18 = None


class ImageEncoder:
    """Encode one property image set with a pretrained ResNet18 trunk."""

    def __init__(self) -> None:
        if torch is None or resnet18 is None:
            raise RuntimeError("Install torch and torchvision to enable image valuation")
        weights = ResNet18_Weights.DEFAULT
        network = resnet18(weights=weights)
        self.model = torch.nn.Sequential(*list(network.children())[:-1]).eval()
        self.transform = weights.transforms()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def encode_paths(self, paths: list[Path]) -> np.ndarray:
        if not paths:
            raise ValueError("At least one image is required for multimodal valuation")
        tensors = []
        for path in paths:
            with Image.open(path).convert("RGB") as image:
                tensors.append(self.transform(image))
        batch = torch.stack(tensors).to(self.device)
        with torch.inference_mode():
            embeddings = self.model(batch).flatten(1).cpu().numpy()
        return embeddings.mean(axis=0).astype(np.float32)

    def encode_bytes(self, values: list[bytes]) -> np.ndarray:
        paths = []
        for value in values:
            image = Image.open(io.BytesIO(value)).convert("RGB")
            tensors = [self.transform(image)]
            batch = torch.stack(tensors).to(self.device)
            with torch.inference_mode():
                paths.append(self.model(batch).flatten(1).cpu().numpy()[0])
        if not paths:
            raise ValueError("At least one image is required for multimodal valuation")
        return np.mean(paths, axis=0).astype(np.float32)


class LateFusionRegressor:
    """Fuse structured property fields and image embeddings into price."""

    def __init__(self) -> None:
        if Ridge is None or OneHotEncoder is None or StandardScaler is None:
            raise RuntimeError("Install scikit-learn to train or use the fusion regressor")
        self.zip_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=10.0)
        self.feature_names: list[str] = []

    @staticmethod
    def _structured_frame(rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
        numeric = np.array([
            [float(row.get("bedrooms", 0)), float(row.get("bathrooms", 0)), float(row.get("area", 0))]
            for row in rows
        ], dtype=np.float32)
        return numeric, ["bedrooms", "bathrooms", "area"]

    def _features(self, rows: list[dict[str, Any]], embeddings: np.ndarray, fit: bool = False) -> np.ndarray:
        numeric, names = self._structured_frame(rows)
        zips = np.array([[str(row.get("zipcode", "unknown"))] for row in rows])
        zip_values = self.zip_encoder.fit_transform(zips) if fit else self.zip_encoder.transform(zips)
        if fit:
            self.feature_names = names + [f"zipcode:{value}" for value in self.zip_encoder.categories_[0]] + [f"image:{i}" for i in range(embeddings.shape[1])]
        return self.scaler.fit_transform(np.hstack([numeric, zip_values, embeddings])) if fit else self.scaler.transform(np.hstack([numeric, zip_values, embeddings]))

    def fit(self, rows: list[dict[str, Any]], embeddings: np.ndarray, prices: np.ndarray) -> "LateFusionRegressor":
        features = self._features(rows, embeddings, fit=True)
        self.model.fit(features, np.log1p(prices))
        return self

    def predict(self, row: dict[str, Any], embedding: np.ndarray) -> float:
        features = self._features([row], embedding.reshape(1, -1), fit=False)
        return float(np.expm1(self.model.predict(features)[0]))

    def explain(self, row: dict[str, Any], embedding: np.ndarray) -> dict[str, float]:
        features = self._features([row], embedding.reshape(1, -1), fit=False)[0]
        coefficients = self.model.coef_
        image_start = len(coefficients) - embedding.shape[0]
        return {
            "structured_signal": float(np.abs(features[:image_start] * coefficients[:image_start]).sum()),
            "image_signal": float(np.abs(features[image_start:] * coefficients[image_start:]).sum()),
        }


def decode_data_url(value: str) -> bytes:
    if "," not in value:
        raise ValueError("Image data must be a data URL")
    return base64.b64decode(value.split(",", 1)[1])
