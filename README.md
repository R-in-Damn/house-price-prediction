# House Price Prediction — End‑to‑End (Ames Housing)

**Technologies:** Python 3.8+, pandas, numpy, matplotlib, seaborn, scikit-learn, SHAP, Streamlit  
**Dataset:** Ames Housing (from Kaggle: *House Prices — Advanced Regression Techniques*).

> This project mirrors the pipeline in your report: preprocessing → EDA → feature engineering → model training (Linear Regression, Random Forest, Gradient Boosting) → GridSearchCV tuning (Gradient Boosting) → explainability with SHAP → deployment via Streamlit.

---

## 1) Quick Start

### A. Create environment & install dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### B. Get the data
Download **train.csv** and **test.csv** from Kaggle (Ames Housing) and put them in:
```
house_price_project/data/
```
Create the folder if needed.

### C. Train models
```bash
python -m src.train_model --data_dir data --target SalePrice
```
This will:
- Clean/encode features
- Train **LinearRegression**, **RandomForestRegressor**, **GradientBoostingRegressor**
- Tune **GradientBoosting** with **GridSearchCV**
- Evaluate on a hold‑out set
- Save the best end‑to‑end pipeline to `artifacts/model_pipeline.joblib`
- Save metrics to `artifacts/metrics.json`

### D. Run the Streamlit app
```bash
streamlit run src/app_streamlit.py
```
Open the shown URL to use the UI: input a few property details → get **real‑time predicted price** + **SHAP explanation**.

---

## 2) Project Structure

```
house_price_project/
├── artifacts/
│   ├── model_pipeline.joblib          # saved end-to-end pipeline (preprocessor + model)
│   └── metrics.json                   # train/test metrics
├── data/                              # put Kaggle files here (train.csv, test.csv)
├── src/
│   ├── train_model.py                 # training & tuning script
│   ├── app_streamlit.py               # deployment UI with SHAP
│   └── utils.py                       # helpers: preprocessing, EDA snippets
├── requirements.txt
└── README.md
```

---

## 3) Notes

- We **focus on a high‑signal subset** of Ames features for the app:
  - Numerical: `LotArea, GrLivArea, TotalBsmtSF, FirstFlrSF, GarageCars, FullBath, YearBuilt`
  - Categorical: `Neighborhood, HouseStyle, KitchenQual, ExterQual`
- The training script uses exactly these columns to keep training and app in sync (you can extend easily).
- SHAP is computed with a **TreeExplainer** for the tuned Gradient Boosting model.
- Metrics include **MAE**, **RMSE**, **R²** for both validation and test splits.

---

## 4) Troubleshooting
- If SHAP plot doesn’t render on some systems, ensure `matplotlib` is up to date and re‑run.
- If memory is low during GridSearch, reduce the grid in `train_model.py` (`param_grid_gbr`).

---

**Name: Arindam, Roll No: 2338654**
