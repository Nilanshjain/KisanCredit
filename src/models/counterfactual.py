"""Counter-factual explanation: "what would need to change for approval?"

Greedy search over the *actionable form fields* an applicant realistically
controls (monthly income, requested loan amount). Each candidate value is
re-mapped through the full v2 feature pipeline before scoring, so every
derived feature (income-to-credit ratio, log income, annuity ratios, …) stays
internally consistent — perturbing the final 149-feature vector directly
would not.

Output powers the "How to improve" card on the application detail page,
narrated into plain language by src/llm/gemini_explainer.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from ..features.home_credit_features import features_from_simple_form
from .predictor import APPROVE_THRESHOLD, ProfitabilityPredictor


@dataclass
class ActionableField:
    field: str               # key in the simple-application form
    label: str               # human label for the UI
    unit: str                # display unit prefix
    higher_is_better: bool   # sweep direction toward approval
    floor: float             # smallest realistic value
    ceiling: float           # largest realistic value


# The applicant genuinely controls these. monthly_expenses is intentionally
# excluded — it has no analogue in the Home Credit feature space, so changing
# it wouldn't move the v2 score and suggesting it would be misleading.
ACTIONABLE_FIELDS: List[ActionableField] = [
    ActionableField("monthly_income", "Monthly income", "₹", higher_is_better=True,  floor=10_000, ceiling=300_000),
    ActionableField("loan_amount",    "Loan amount",    "₹", higher_is_better=False, floor=50_000, ceiling=5_000_000),
]

MAX_CHANGES = 2          # narrate at most this many adjustments
SWEEP_STEPS = 12         # candidate values tested per field
MIN_USEFUL_LIFT = 0.02   # ignore changes that move the score < this


def _predict_from_form(
    predictor: ProfitabilityPredictor, feature_names: List[str], form: Dict[str, Any],
) -> float:
    """Map a form dict onto v2 features and return the model's score."""
    feats = features_from_simple_form(form, feature_names)
    return float(predictor.predict(pd.DataFrame([feats]), return_confidence=False))


def _best_change_for_field(
    predictor: ProfitabilityPredictor,
    feature_names: List[str],
    form: Dict[str, Any],
    af: ActionableField,
) -> Optional[Dict[str, Any]]:
    """Sweep one form field toward improvement; return the most score-lifting value."""
    current = float(form.get(af.field, 0) or 0)
    if current <= 0:
        return None

    if af.higher_is_better:
        top = min(max(current * 1.8, current + 1), af.ceiling)
        candidates = [current + (top - current) * i / SWEEP_STEPS for i in range(1, SWEEP_STEPS + 1)]
    else:
        bottom = max(min(current * 0.5, current - 1), af.floor)
        candidates = [current - (current - bottom) * i / SWEEP_STEPS for i in range(1, SWEEP_STEPS + 1)]

    base_score = _predict_from_form(predictor, feature_names, form)
    best: Optional[Dict[str, Any]] = None
    for val in candidates:
        trial = dict(form)
        trial[af.field] = round(float(val), 0)
        score = _predict_from_form(predictor, feature_names, trial)
        lift = score - base_score
        if lift < MIN_USEFUL_LIFT:
            continue
        if best is None or lift > best["delta_score"]:
            best = {
                "feature": af.field,
                "display_label": af.label,
                "display_unit": af.unit,
                "current": round(current, 0),
                "suggested": round(float(val), 0),
                "delta_score": round(lift, 4),
                "new_score": round(score, 4),
            }
    return best


def find_counterfactual(
    predictor: ProfitabilityPredictor,
    simple_form: Dict[str, Any],
    feature_names: List[str],
    target_threshold: float = APPROVE_THRESHOLD,
    max_changes: int = MAX_CHANGES,
) -> Dict[str, Any]:
    """Greedy minimum-change search to push the score above `target_threshold`.

    Returns
    -------
    {
      "reachable": bool,            # did we cross the threshold?
      "starting_score": float,
      "final_score": float,
      "changes": [ {feature, display_label, display_unit, current,
                    suggested, delta_score, new_score}, ... ],
    }
    """
    working = dict(simple_form)
    start_score = _predict_from_form(predictor, feature_names, working)

    if start_score >= target_threshold:
        return {
            "reachable": True,
            "starting_score": round(start_score, 4),
            "final_score": round(start_score, 4),
            "changes": [],
        }

    changes: List[Dict[str, Any]] = []
    tried: set[str] = set()
    for _ in range(max_changes):
        candidates: List[Dict[str, Any]] = []
        for af in ACTIONABLE_FIELDS:
            if af.field in tried:
                continue
            best = _best_change_for_field(predictor, feature_names, working, af)
            if best is not None:
                candidates.append(best)
        if not candidates:
            break

        candidates.sort(key=lambda c: c["delta_score"], reverse=True)
        chosen = candidates[0]
        changes.append(chosen)
        working[chosen["feature"]] = chosen["suggested"]
        tried.add(chosen["feature"])
        if chosen["new_score"] >= target_threshold:
            break

    final_score = _predict_from_form(predictor, feature_names, working)
    return {
        "reachable": final_score >= target_threshold,
        "starting_score": round(start_score, 4),
        "final_score": round(final_score, 4),
        "changes": changes,
    }
