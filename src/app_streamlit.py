
# Purpose: Streamlit app for real-time price prediction with SHAP explanations.
# Name: Arindam, Roll No: 2338654

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from joblib import load

import shap
from sklearn.ensemble import GradientBoostingRegressor

# ---- UI ----
st.set_page_config(page_title="House Price Prediction (Ames)", layout="centered")

st.title("🏠 House Price Prediction — Ames")
st.caption("Interactive ML app with SHAP explainability (Gradient Boosting).")
st.write("**Name: Arindam, Roll No: 2338654**")

# Load pipeline
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "model_pipeline.joblib")
if not os.path.exists(MODEL_PATH):
    st.error("Model pipeline not found. Please run training first (see README).")
    st.stop()

pipe = load(MODEL_PATH)

# The pipeline was trained on this fixed set of features:
NUM_FEATURES = ["LotArea", "GrLivArea", "TotalBsmtSF", "1stFlrSF", "GarageCars", "FullBath", "YearBuilt"]
CAT_FEATURES = ["Neighborhood", "HouseStyle", "KitchenQual", "ExterQual"]

# Some helpful defaults for select boxes (safe fallbacks if not in training data)
neighborhoods = ["NAmes","CollgCr","OldTown","Edwards","Somerst","NridgHt","Sawyer","Gilbert","NWAmes"]
housestyles = ["1Story","2Story","1.5Fin","SLvl","SFoyer","2.5Unf"]
qual_levels = ["Ex","Gd","TA","Fa"]

with st.form("inputs"):
    st.subheader("Enter Property Details")
    c1, c2 = st.columns(2)
    with c1:
        lotarea = st.number_input("LotArea (sq ft)", min_value=1000, max_value=100000, value=8450)
        grliv = st.number_input("GrLivArea (Above grade living area, sq ft)", min_value=200, max_value=6000, value=1460)
        bsmt = st.number_input("TotalBsmtSF (sq ft)", min_value=0, max_value=4000, value=856)
        firstflr = st.number_input("1stFlrSF (sq ft)", min_value=200, max_value=4000, value=856)
    with c2:
        garage = st.number_input("GarageCars", min_value=0, max_value=4, value=2)
        fullbath = st.number_input("FullBath", min_value=0, max_value=4, value=2)
        yearbuilt = st.number_input("YearBuilt", min_value=1870, max_value=2025, value=2003)

    c3, c4 = st.columns(2)
    with c3:
        neighborhood = st.selectbox("Neighborhood", neighborhoods, index=0)
        housestyle = st.selectbox("HouseStyle", housestyles, index=0)
    with c4:
        kitchenqual = st.selectbox("KitchenQual", qual_levels, index=2)
        exterqual = st.selectbox("ExterQual", qual_levels, index=2)

    submitted = st.form_submit_button("Predict Price")

if submitted:
    X = pd.DataFrame([{
        "LotArea": lotarea,
        "GrLivArea": grliv,
        "TotalBsmtSF": bsmt,
        "1stFlrSF": firstflr,
        "GarageCars": garage,
        "FullBath": fullbath,
        "YearBuilt": yearbuilt,
        "Neighborhood": neighborhood,
        "HouseStyle": housestyle,
        "KitchenQual": kitchenqual,
        "ExterQual": exterqual
    }])

    pred = float(pipe.predict(X)[0])
    st.success(f"**Predicted Sale Price:** ${pred:,.0f}")

    # ---- SHAP Explainability (bar chart for the single instance) ----
    # Try to get the underlying model for TreeExplainer
    try:
        model = pipe.named_steps["model"]
    except Exception:
        model = None

    st.subheader("Why this prediction? (SHAP)")
    try:
        if hasattr(model, "predict") and isinstance(model, GradientBoostingRegressor):
            # Build a background sample from the current input (small baseline)
            background = X.copy()
            explainer = shap.TreeExplainer(model)
            # Transform X using the preprocessor to match model input
            pre = pipe.named_steps["pre"]
            X_trans = pre.transform(X)
            shap_vals = explainer.shap_values(X_trans)

            # Aggregate SHAP values by original semantic features:
            # This is approximate when OHE is present; for simplicity we show top-level numeric features
            # and show one contribution per original feature by summing OHE columns.
            # We'll map back numeric features directly.
            contrib = {}

            # For numeric features: they are first in ColumnTransformer
            # Get feature names from preprocessor where available
            try:
                num_names = pre.transformers_[0][2]
                cat_encoder = pre.transformers_[1][1].named_steps["ohe"]
                cat_feature_names = cat_encoder.get_feature_names_out(pre.transformers_[1][2])
            except Exception:
                num_names = NUM_FEATURES
                cat_feature_names = []

            import numpy as np
            shap_row = np.array(shap_vals)[0]  # shape (n_transformed_features,)

            # Figure out index splits
            num_len = len(num_names)
            num_contrib = shap_row[:num_len]
            cat_contrib = shap_row[num_len:]
            for i, n in enumerate(num_names):
                contrib[n] = float(num_contrib[i])
            # Sum categorical OHE contributions back to original feature buckets
            for full in cat_feature_names:
                base = full.split("_")[0]
                contrib[base] = contrib.get(base, 0.0) + float(cat_contrib[0])
                cat_contrib = cat_contrib[1:]

            # Plot
            items = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
            labels = [k for k, _ in items]
            values = [v for _, v in items]

            fig, ax = plt.subplots()
            ax.barh(labels[::-1], values[::-1])
            ax.set_xlabel("SHAP value (impact on prediction)")
            ax.set_ylabel("Feature")
            st.pyplot(fig)
        else:
            st.info("SHAP visualization is only available for the tuned Gradient Boosting model.")
    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")

st.caption("Name: Arindam, Roll No: 2338654")

# Footer: Name: Arindam, Roll No: 2338654
