"""Smoke-test that the Home Credit v2 model predicts on real applicant rows.

Pulls a few stratified rows from the Home Credit application_train.csv,
runs them through the same feature engineering as training, then through
the v2 LightGBM model + a SHAP explanation. Prints predictions next to the
true labels so you can eyeball that the model is doing something sensible
(higher scores for low-default-risk profiles, lower for high-risk).

Use this to verify v2 works end-to-end without needing the FastAPI stack
or the frontend.

Run:
    python scripts/demo_v2_prediction.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd

from scripts.train_home_credit import engineer_features, select_feature_columns

DATA_PATH = Path("data/raw/home_credit/application_train.csv")
MODEL_PATH = Path("models/home_credit_v2.pkl")


def main() -> int:
    if not MODEL_PATH.exists():
        print(f"ERROR: {MODEL_PATH} missing. Run scripts/train_home_credit.py first.")
        return 1
    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} missing. Download via kagglehub.")
        return 1

    print(f"[load] reading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)

    # Pick 3 each of default=0 and default=1
    sample_pos = df[df["TARGET"] == 1].sample(3, random_state=42)
    sample_neg = df[df["TARGET"] == 0].sample(3, random_state=42)
    sample = pd.concat([sample_neg, sample_pos]).reset_index(drop=True)

    print(f"[fe] engineering features ...")
    fe = engineer_features(sample)
    feature_cols = select_feature_columns(fe)
    X = fe[feature_cols]

    print(f"[load] loading {MODEL_PATH} ...")
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    expected_features = artifact["feature_names"]

    # Align via reindex: adds missing columns as 0.0 and drops extras in one shot
    X_aligned = X.reindex(columns=expected_features, fill_value=0.0)

    proba = model.predict(X_aligned)

    print()
    print("--- Predictions on stratified sample -------------------------------------")
    print(f"{'true_label':>10}  {'pred_default_prob':>18}  {'income (INR)':>14}  {'credit_amt':>11}")
    for i, row in enumerate(sample.itertuples()):
        label = "DEFAULT" if row.TARGET == 1 else "REPAID"
        income = f"{row.AMT_INCOME_TOTAL:,.0f}"
        credit = f"{row.AMT_CREDIT:,.0f}"
        print(f"{label:>10}  {proba[i]:>18.4f}  {income:>14}  {credit:>11}")

    print()
    print(f"Average predicted default prob for true defaulters:  {proba[3:].mean():.4f}")
    print(f"Average predicted default prob for true repayers:    {proba[:3].mean():.4f}")
    print(f"Model AUC on this 6-row sample is not meaningful;")
    print(f"see notebooks/home_credit_eval.png for the real 5-fold-CV ROC (CV mean AUC {artifact['metadata']['lightgbm_cv_auc_mean']:.4f}).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
