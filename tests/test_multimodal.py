from pathlib import Path

import numpy as np

from backend.app.multimodal import LateFusionRegressor
from backend.app.services import RealEstateService


def test_late_fusion_regressor_combines_structured_and_image_features():
    rows = [
        {"bedrooms": 2, "bathrooms": 1, "area": 1000, "zipcode": "A"},
        {"bedrooms": 4, "bathrooms": 3, "area": 2200, "zipcode": "B"},
        {"bedrooms": 3, "bathrooms": 2, "area": 1800, "zipcode": "A"},
    ]
    embeddings = np.arange(3 * 8, dtype=np.float32).reshape(3, 8)
    model = LateFusionRegressor().fit(rows, embeddings, np.array([300000, 650000, 500000]))
    prediction = model.predict(rows[0], embeddings[0])
    explanation = model.explain(rows[0], embeddings[0])
    assert prediction > 0
    assert set(explanation) == {"structured_signal", "image_signal"}


def test_archive_image_resolution_is_sandboxed(tmp_path: Path):
    archive = tmp_path / "Houses Dataset"
    archive.mkdir()
    service = RealEstateService(archive_dir=archive, model_path=tmp_path / "missing.joblib")
    inside = archive / "1_frontal.jpg"
    inside.write_bytes(b"not-used-by-this-test")
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"not-used-by-this-test")
    resolved = service._resolve_image_paths([str(inside), str(outside)])
    assert resolved == [inside.resolve()]
