"""Home Credit feature engineering — shared by training and serving.

The v2 model is deliberately trained on a *restricted* feature set: only
features an applicant can supply at application time. No EXT_SOURCE bureau
scores, no building/document statistics — the whole product premise is
scoring thin-file borrowers who lack a bureau record, so the model must not
depend on one.

`engineer_features`        derives features from raw Home Credit columns.
`select_serving_features`  the canonical column set — used by BOTH the
                           training script and the live inference path so the
                           model is scored on exactly the features it was
                           trained on (no train/serve skew, no zero-fill).
`features_from_simple_form` maps the frontend form onto that feature vector.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd


# ─────────────────────── core feature engineering ───────────────────────


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive model features from raw Home Credit columns.

    Defensive about missing columns so it runs on both the full training
    table and a single synthesised inference row.
    """
    out = df.copy()

    # income
    out["income_monthly_avg"] = out["AMT_INCOME_TOTAL"] / 12.0
    out["income_to_credit_ratio"] = out["AMT_INCOME_TOTAL"] / (out["AMT_CREDIT"] + 1.0)
    out["income_to_annuity_ratio"] = out["AMT_INCOME_TOTAL"] / (out["AMT_ANNUITY"] + 1.0)
    out["income_per_family_member"] = out["AMT_INCOME_TOTAL"] / (out["CNT_FAM_MEMBERS"] + 1.0)
    out["income_log"] = np.log1p(out["AMT_INCOME_TOTAL"])

    # debt burden
    out["annuity_to_income_ratio"] = out["AMT_ANNUITY"] / (out["AMT_INCOME_TOTAL"] + 1.0)
    out["credit_to_goods_ratio"] = out["AMT_CREDIT"] / (out["AMT_GOODS_PRICE"].fillna(0) + 1.0)
    out["credit_log"] = np.log1p(out["AMT_CREDIT"])

    # employment / age
    out["days_employed_safe"] = out["DAYS_EMPLOYED"].replace(365243, np.nan).abs()
    out["days_birth_years"] = (out["DAYS_BIRTH"].abs() / 365.0).round(1)
    out["employment_ratio"] = out["days_employed_safe"] / out["DAYS_BIRTH"].abs().replace(0, np.nan)

    # ownership flags — Home Credit stores these as 'Y'/'N' strings
    for flag in ("FLAG_OWN_CAR", "FLAG_OWN_REALTY"):
        if flag in out.columns and out[flag].dtype == object:
            out[flag] = (out[flag] == "Y").astype(int)

    # categoricals the applicant form collects — one-hot encoded
    categorical_cols = [
        c for c in ("CODE_GENDER", "NAME_EDUCATION_TYPE", "NAME_HOUSING_TYPE", "NAME_INCOME_TYPE")
        if c in out.columns
    ]
    if categorical_cols:
        out = pd.get_dummies(out, columns=categorical_cols, drop_first=True, dummy_na=False)

    return out


# Raw + engineered numeric columns the model uses. Every one is derivable from
# the applicant form — no bureau scores, no building/document statistics.
SERVING_NUMERIC_FEATURES: List[str] = [
    # raw
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "CNT_CHILDREN", "CNT_FAM_MEMBERS", "DAYS_BIRTH", "DAYS_EMPLOYED",
    "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    # engineered
    "income_monthly_avg", "income_to_credit_ratio", "income_to_annuity_ratio",
    "income_per_family_member", "income_log", "annuity_to_income_ratio",
    "credit_to_goods_ratio", "credit_log", "days_employed_safe",
    "days_birth_years", "employment_ratio",
]

# One-hot columns matching any of these prefixes are kept too.
SERVING_CATEGORICAL_PREFIXES = (
    "CODE_GENDER_", "NAME_EDUCATION_TYPE_", "NAME_HOUSING_TYPE_", "NAME_INCOME_TYPE_",
)


def select_serving_features(df: pd.DataFrame) -> List[str]:
    """The canonical feature column set — restricted, no bureau data.

    Used by the training script and the inference path alike, so the model is
    trained and served on identical features.
    """
    cols = [c for c in SERVING_NUMERIC_FEATURES if c in df.columns]
    cols += [
        c for c in df.columns
        if c.startswith(SERVING_CATEGORICAL_PREFIXES) and c not in cols
    ]
    return cols


# Back-compat alias — the training script previously called this name.
def select_feature_columns(df: pd.DataFrame) -> List[str]:
    return select_serving_features(df)


# ─────────────────── form → Home Credit row (serving) ────────────────────


def _age_days_from_dob(dob: str) -> int:
    """Negative day-count from a YYYY-MM-DD date of birth (Home Credit convention)."""
    try:
        born = datetime.strptime(dob, "%Y-%m-%d")
        return -int((datetime.utcnow() - born).days)
    except Exception:
        return -int(35 * 365.25)


def build_home_credit_row(form: Dict) -> pd.DataFrame:
    """One raw Home Credit-shaped row from the applicant form."""
    income = float(form.get("monthly_income", 0) or 0)
    loan_amount = float(form.get("loan_amount", 0) or 0)
    dependents = int(form.get("dependents", 0) or 0)
    employment_years = float(form.get("employment_years", 0) or 0)

    return pd.DataFrame([{
        "AMT_INCOME_TOTAL": income * 12.0,
        "AMT_CREDIT": loan_amount,
        "AMT_ANNUITY": loan_amount * 0.05,        # ~median annuity/credit ratio
        "AMT_GOODS_PRICE": loan_amount * 0.90,
        "CNT_CHILDREN": dependents,
        "CNT_FAM_MEMBERS": float(dependents + 1),
        "DAYS_BIRTH": _age_days_from_dob(str(form.get("date_of_birth", ""))),
        "DAYS_EMPLOYED": -int(employment_years * 365.25),
        "FLAG_OWN_CAR": 1 if form.get("owns_car") else 0,
        "FLAG_OWN_REALTY": 1 if form.get("owns_property") else 0,
    }])


def features_from_simple_form(form: Dict, feature_names: List[str]) -> Dict[str, float]:
    """Build the v2 feature vector from the applicant form.

    `feature_names` is the model's expected feature list — used to set the
    correct one-hot dummy columns (single-row get_dummies can't reproduce the
    training-time column set).
    """
    engineered = engineer_features(build_home_credit_row(form))

    feats: Dict[str, float] = {}
    for c in select_serving_features(engineered):
        val = engineered[c].iloc[0]
        if c in feature_names and pd.notna(val):
            feats[c] = float(val)

    # Categorical one-hots — set explicitly against the model's columns.
    gender = str(form.get("gender", "")).strip().lower()
    _set_dummy(feats, feature_names, "CODE_GENDER", "M" if gender == "male" else "F")
    _set_dummy(feats, feature_names, "NAME_EDUCATION_TYPE",
               str(form.get("education_level", "Secondary / secondary special")))
    _set_dummy(feats, feature_names, "NAME_HOUSING_TYPE",
               str(form.get("housing_type", "House / apartment")))
    _set_dummy(feats, feature_names, "NAME_INCOME_TYPE",
               str(form.get("employment_type", "Working")))
    return feats


def _set_dummy(feats: Dict[str, float], feature_names: List[str], prefix: str, value: str) -> None:
    """Set the `{prefix}_{value}` one-hot column to 1.0 if the model has it.

    If `value` was the drop_first reference category there is no such column —
    leaving all of this prefix's dummies at 0 correctly encodes that.
    """
    target = f"{prefix}_{value}"
    if target in feature_names:
        feats[target] = 1.0
