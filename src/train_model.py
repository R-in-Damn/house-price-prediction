
# Purpose: Train & tune models on Ames subset; save best pipeline and metrics.
# Name: Arindam, Roll No: 2338654

from __future__ import annotations
import os, json, argparse
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from .utils import build_preprocessor, select_columns, train_test_split_fixed, ALL_FEATURES

def evaluate(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    # Some sklearn builds don’t support the `squared` kwarg — compute RMSE manually
    mse = float(mean_squared_error(y_true, y_pred))   # default is squared=True
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def load_data(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Put Kaggle Ames 'train.csv' (the one with SalePrice) in the data/ folder."
        )
    df = pd.read_csv(path)
    required = set(ALL_FEATURES + ["SalePrice"])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Your data\\train.csv is not the expected Kaggle Ames TRAIN file.\n"
            f"Missing columns: {missing}\n"
            "Make sure you used Kaggle 'train.csv' (not test.csv) and placed it in the data/ folder."
        )
    return df[list(required)].copy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--target", type=str, default="SalePrice")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    args = parser.parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)

    df = load_data(args.data_dir)
    X_train, X_test, y_train, y_test = train_test_split_fixed(df, target=args.target)

    pre = build_preprocessor()

    # Base models
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=42)
    }

    results = {}
    best_name, best_score = None, np.inf
    best_pipeline = None

    for name, model in models.items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(X_train, y_train)
        pred_val = pipe.predict(X_test)
        metrics = evaluate(y_test, pred_val)
        results[name] = metrics
        if metrics["RMSE"] < best_score:
            best_name, best_score, best_pipeline = name, metrics["RMSE"], pipe

    # Hyperparameter tuning for Gradient Boosting
    param_grid_gbr = {
        "model__n_estimators": [200, 400],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2, 3],
        "model__subsample": [0.8, 1.0]
    }

    gbr_pipe = Pipeline([("pre", pre), ("model", GradientBoostingRegressor(random_state=42))])
    grid = GridSearchCV(
        gbr_pipe, param_grid=param_grid_gbr,
        scoring="neg_root_mean_squared_error", cv=5, n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)

    tuned_pred = grid.best_estimator_.predict(X_test)
    tuned_metrics = evaluate(y_test, tuned_pred)
    results["GradientBoosting_Tuned"] = tuned_metrics

    # Choose final best
    final_estimator = grid.best_estimator_ if tuned_metrics["RMSE"] < best_score else best_pipeline
    final_name = "GradientBoosting_Tuned" if tuned_metrics["RMSE"] < best_score else best_name

    # Save artifacts
    dump(final_estimator, os.path.join(args.artifacts_dir, "model_pipeline.joblib"))
    with open(os.path.join(args.artifacts_dir, "metrics.json"), "w") as f:
        json.dump({"model": final_name, "results": results}, f, indent=2)

    print("Training complete. Best model:", final_name)
    print("Metrics:", json.dumps(results, indent=2))

if __name__ == "__main__":
    main()

# Footer: Name: Arindam, Roll No: 2338654
