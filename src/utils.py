
# Purpose: Utility helpers for preprocessing and EDA for the Ames Housing project.
# Name: Arindam, Roll No: 2338654

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, RobustScaler
from sklearn.impute import SimpleImputer

# ---- Feature schema used by both training and the Streamlit app ----
NUM_FEATURES = [
    "LotArea", "GrLivArea", "TotalBsmtSF", "1stFlrSF", "GarageCars",
    "FullBath", "YearBuilt"
]
CAT_FEATURES = [
    "Neighborhood", "HouseStyle", "KitchenQual", "ExterQual"
]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the selected feature columns, creating any that are missing."""
    out = df.copy()
    for c in ALL_FEATURES:
        if c not in out.columns:
            out[c] = np.nan
    return out[ALL_FEATURES]

def _log1p_safe(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(x, 0))

def build_preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler(with_centering=True))
    ])

    cat_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    pre = ColumnTransformer(transformers=[
        ("num", num_pipe, NUM_FEATURES),
        ("cat", cat_pipe, CAT_FEATURES)
    ])
    return pre


def train_test_split_fixed(
    df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    from sklearn.model_selection import train_test_split
    X = select_columns(df)
    y = df[target].astype(float)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Footer: Name: Arindam, Roll No: 2338654
