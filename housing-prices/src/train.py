# src/train.py
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
import sklearn

ROOT = r"F:\Data Science\housing-prices"
DATA_PATH = os.path.join(ROOT, "data", "train.csv")
OUT_SIMPLE = os.path.join(ROOT, "models", "final_model.joblib")
OUT_TUNED = os.path.join(ROOT, "models", "final_model_tuned.joblib")

print("sklearn version:", sklearn.__version__)
assert os.path.exists(DATA_PATH), f"Training CSV not found at {DATA_PATH}"

df = pd.read_csv(DATA_PATH)
df['log_SalePrice'] = np.log1p(df['SalePrice'])

# features / target
X = df.drop(columns=['Id', 'SalePrice', 'log_SalePrice'])
y = df['log_SalePrice']

# feature lists
num_cols = X.select_dtypes(include=['number']).columns.tolist()
cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
print(f"Num cols: {len(num_cols)}, Cat cols: {len(cat_cols)}")

# robust OneHotEncoder to support different sklearn versions
try:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("ohe", ohe)
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
], remainder="drop")

# quick pipeline (no tuning)
pipe = Pipeline([
    ("preproc", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("Fitting quick pipeline (n_estimators=100)...")
pipe.fit(X_train, y_train)

# evaluate
preds_log = pipe.predict(X_test)
preds_price = np.expm1(preds_log)
y_test_price = np.expm1(y_test)
mae = mean_absolute_error(y_test_price, preds_price)
print(f"Quick pipeline MAE ($): {mae:,.2f}")

# save simple pipeline
os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
joblib.dump(pipe, OUT_SIMPLE)
print("Saved simple pipeline to:", OUT_SIMPLE)

# OPTIONAL: small randomized tuning (set run_tune=True to enable)
run_tune = False
if run_tune:
    print("Starting small RandomizedSearchCV tuning (this may take a while)...")
    param_dist = {
        'model__n_estimators': [100, 200, 400],
        'model__max_depth': [None, 10, 20, 30],
        'model__max_features': ['sqrt', 'log2', 0.3, 0.5]
    }
    pipe_for_tune = Pipeline([
        ("preproc", preprocessor),
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
    ])
    rs = RandomizedSearchCV(pipe_for_tune, param_distributions=param_dist, n_iter=6, cv=3,
                            scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42, verbose=2)
    rs.fit(X_train, y_train)
    print("Best CV MAE (log):", -rs.best_score_)
    best_pipe = rs.best_estimator_
    joblib.dump(best_pipe, OUT_TUNED)
    print("Saved tuned pipeline to:", OUT_TUNED)