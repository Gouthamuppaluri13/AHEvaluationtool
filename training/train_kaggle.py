"""
Train an online scikit-learn model bundle from Kaggle.

Default profile: justinas/startup-success-prediction
- Target column: status (string); mapped to binary success label

Usage examples:
  python -m models.training.train_kaggle --dataset-ref justinas/startup-success-prediction --target-col status
  python -m models.training.train_kaggle --dataset-ref justinas/startup-success-prediction --csv-name startups.csv --target-col status

Env:
  KAGGLE_USERNAME, KAGGLE_KEY

Output:
  models/startup_model.joblib
"""
import os
import argparse
import tempfile
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

def download_kaggle_dataset(dataset_ref: str, out_dir: str):
    import subprocess
    cmd = ["kaggle", "datasets", "download", "-d", dataset_ref, "-p", out_dir, "--unzip"]
    subprocess.check_call(cmd)

def build_features_justinas(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    df = df.copy()
    if "founded_at" in df.columns:
        try:
            df["founded_year"] = pd.to_datetime(df["founded_at"], errors="coerce").dt.year
        except Exception:
            pass
    if "founded_year" not in df.columns:
        df["founded_year"] = df.get("founded_year", pd.Series([2018]*len(df)))

    df["age"] = 2025 - df["founded_year"].fillna(2018).astype(float)

    cats = []
    for c in ["category_code", "country_code", "state_code", "region", "category_list", "market", "status", "stage"]:
        if c in df.columns:
            cats.append(c)

    nums = ["age"]
    X = df[nums + cats].copy()
    X = X.fillna(0)

    y = df[target_col]
    if y.dtype == object:
        y = y.astype(str).str.lower().str.contains("operating|acquired|ipo|successful|success").astype(int)
    else:
        y = (y > y.median()).astype(int)

    return X, y, nums, cats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-ref", default="justinas/startup-success-prediction")
    ap.add_argument("--csv-name", required=False)
    ap.add_argument("--target-col", default="status")
    ap.add_argument("--out", default="models/startup_model.joblib")
    args = ap.parse_args()

    os.makedirs("models", exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        download_kaggle_dataset(args.dataset_ref, td)
        if args.csv_name:
            path = os.path.join(td, args.csv_name)
        else:
            csvs = [os.path.join(td, f) for f in os.listdir(td) if f.endswith(".csv")]
            path = max(csvs, key=lambda p: os.path.getsize(p))
        df = pd.read_csv(path)

    X, y, numeric, categoricals = build_features_justinas(df, args.target_col)

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categoricals)
        ],
        remainder="drop"
    )
    clf = Pipeline(steps=[
        ("pre", pre),
        ("model", LogisticRegression(max_iter=500))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() == 2 else None
    )
    clf.fit(X_train, y_train)
    auc = None
    try:
        proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
    except Exception:
        pass

    bundle = {
        "clf": clf,
        "reg": None,
        "feature_list": list(X.columns),
        "categoricals": categoricals,
        "meta": {"source": args.dataset_ref, "auc": auc}
    }
    import joblib
    joblib.dump(bundle, args.out)
    print(f"Saved model bundle to {args.out} (AUC={auc})")

if __name__ == "__main__":
    main()
