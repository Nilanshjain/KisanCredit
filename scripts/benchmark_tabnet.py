"""Benchmark: TabNet (PyTorch deep learning) vs LightGBM on Home Credit.

Trains a TabNet classifier — an attention-based deep-learning architecture for
tabular data (Arik & Pfister, 2019), implemented in PyTorch — on the *same*
features and *same* stratified CV folds as the production LightGBM model.

The point is the comparison, not a new production model: it shows the model
choice (gradient boosting) was evidence-based. On tabular data at this scale,
GBMs typically beat deep learning — this quantifies by how much.

Run:
    python scripts/benchmark_tabnet.py

Outputs notebooks/tabnet_comparison.json with both models' CV AUC.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.features.home_credit_features import engineer_features, select_serving_features

DATA_PATH = Path("data/raw/home_credit/application_train.csv")
OUT_PATH = Path("notebooks/tabnet_comparison.json")
RANDOM_STATE = 42
N_FOLDS = 5


def main() -> int:
    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} missing — download Home Credit first.", file=sys.stderr)
        return 1

    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        import torch
    except ImportError:
        print("ERROR: pip install pytorch-tabnet torch", file=sys.stderr)
        return 1

    print(f"[load] {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    fe = engineer_features(df)
    feature_cols = select_serving_features(fe)
    X = fe[feature_cols].copy()
    y = df["TARGET"].astype(int).to_numpy()

    # TabNet needs dense, finite input — impute NaN with the column median.
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    X = X.to_numpy(dtype=np.float32)
    print(f"[shape] X={X.shape}  positive_rate={y.mean():.4f}  device={'cuda' if torch.cuda.is_available() else 'cpu'}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs: list[float] = []
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        t0 = time.time()
        clf = TabNetClassifier(
            n_d=16, n_a=16, n_steps=4, gamma=1.4,
            seed=RANDOM_STATE, verbose=0,
        )
        clf.fit(
            X[tr], y[tr],
            eval_set=[(X[va], y[va])],
            eval_metric=["auc"],
            max_epochs=60, patience=12, batch_size=8192, virtual_batch_size=1024,
        )
        proba = clf.predict_proba(X[va])[:, 1]
        fold_auc = roc_auc_score(y[va], proba)
        aucs.append(fold_auc)
        print(f"[tabnet] fold {fold}/{N_FOLDS}  AUC={fold_auc:.4f}  ({time.time() - t0:.0f}s)")

    tabnet_auc = float(np.mean(aucs))
    print(f"[tabnet] CV mean AUC = {tabnet_auc:.4f} ± {np.std(aucs):.4f}")

    # Pull the LightGBM number from the production artifact for the comparison.
    lgbm_auc = None
    try:
        import joblib
        art = joblib.load("models/home_credit_v2.pkl")
        lgbm_auc = art["metadata"].get("lightgbm_cv_auc_mean")
    except Exception:
        pass

    result = {
        "dataset": "home-credit-default-risk",
        "n_features": len(feature_cols),
        "cv_folds": N_FOLDS,
        "tabnet_cv_auc": round(tabnet_auc, 4),
        "tabnet_cv_auc_std": round(float(np.std(aucs)), 4),
        "lightgbm_cv_auc": lgbm_auc,
        "winner": "lightgbm" if (lgbm_auc and lgbm_auc >= tabnet_auc) else "tabnet",
        "auc_delta": round((lgbm_auc - tabnet_auc), 4) if lgbm_auc else None,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"\n[done] {json.dumps(result, indent=2)}")
    print(f"[done] saved {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
