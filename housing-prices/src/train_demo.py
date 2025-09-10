# src/train_demo.py
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

ROOT = r"F:\Data Science\housing-prices"
DATA_PATH = os.path.join(ROOT, "data", "train.csv")
OUT_MODEL = os.path.join(ROOT, "models", "final_model.joblib")

print("Loading data...")
df = pd.read_csv(DATA_PATH)
df["log_SalePrice"] = np.log1p(df["SalePrice"])

X = df.drop(columns=["Id", "SalePrice", "log_SalePrice"])
y = df["log_SalePrice"]

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Preprocessing
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("ohe", ohe)
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# ⚡ Train a *smaller* RandomForest (lighter for deployment)
model = RandomForestRegressor(
    n_estimators=50,  # fewer trees = smaller file
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline([
    ("preproc", preprocessor),
    ("model", model)
])

print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(X_train, y_train)

# Evaluate quickly
preds_log = pipe.predict(X_test)
preds_price = np.expm1(preds_log)
y_test_price = np.expm1(y_test)
mae = mean_absolute_error(y_test_price, preds_price)
print(f"Demo model Test MAE: ${mae:,.0f}")

# Save
os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
joblib.dump(pipe, OUT_MODEL)
print(f"Saved demo model to: {OUT_MODEL}")