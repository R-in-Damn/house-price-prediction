from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from joblib import dump

from backend.app.multimodal import ImageEncoder, LateFusionRegressor
from backend.app.services import RealEstateService


def image_paths(directory: Path, property_id: str) -> list[Path]:
    return sorted(directory.glob(f"{property_id}_*.jpg"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a structured plus image late-fusion property model")
    parser.add_argument("--data_dir", default="data/archive (1)/Houses Dataset")
    parser.add_argument("--artifacts_dir", default="artifacts")
    args = parser.parse_args()

    archive = Path(args.data_dir)
    service = RealEstateService(archive_dir=archive, model_path=Path("artifacts/model_pipeline.joblib"))
    encoder = ImageEncoder()
    rows: list[dict[str, object]] = []
    embeddings: list[np.ndarray] = []
    prices: list[float] = []
    for row in service.records.to_dict("records"):
        paths = image_paths(archive, str(row["property_id"]))
        if len(paths) < 1:
            continue
        rows.append(row)
        embeddings.append(encoder.encode_paths(paths))
        prices.append(float(row["price"]))
        print(f"Encoded property {row['property_id']} ({len(rows)}/{len(service.records)})")

    model = LateFusionRegressor().fit(rows, np.vstack(embeddings), np.asarray(prices))
    output = Path(args.artifacts_dir) / "multimodal_fusion.joblib"
    output.parent.mkdir(parents=True, exist_ok=True)
    dump(model, output)
    print(f"Saved {output} using {len(rows)} properties and {embeddings[0].shape[0]} image features")


if __name__ == "__main__":
    main()
