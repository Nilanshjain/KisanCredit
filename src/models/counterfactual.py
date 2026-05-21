"""Counter-factual explanation: "what would need to change for this decision to flip?"

Greedy 1-D search over a small set of *actionable* features (income, expenses,
discipline scores) — perturb each within a realistic range, find the
minimum-change set that lifts the score above the approve threshold. The
output is what powers the "How to improve" card on the user-facing detail page,
narrated by Gemini in src/llm/gemini_explainer.py.

We intentionally restrict the search to a curated whitelist rather than all 45
features because:
- A counter-factual on `ext_sources_mean` is not actionable for a user — they
  can't directly raise an opaque bureau score
- One-hot encoded categoricals (gender, education) shouldn't be suggested
- The Gemini layer narrates 1-2 changes max; brute-forcing all features is
  wasted compute

Algorithm:
1. For each whitelisted feature, sweep N=10 values across a realistic range
2. Take the single change that gives the biggest score lift
3. Apply it, re-predict; if still below threshold, repeat for the next feature
4. Stop after up to MAX_CHANGES adjustments or once approve threshold is hit
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .predictor import APPROVE_THRESHOLD, ProfitabilityPredictor


# Per-feature search ranges + step direction (higher_is_better=True means
# we try INCREASING the value; False means decreasing helps). These are
# expressed against the v1 synthetic feature space; the v2 Home Credit retrain
# will need its own whitelist (Phase 5 task).
# Ranges chosen to be realistic for an Indian rural-applicant context.
@dataclass
class ActionableFeature:
    name: str
    realistic_min: float
    realistic_max: float
    higher_is_better: bool
    display_label: str
    display_unit: str = ""


ACTIONABLE_FEATURES: List[ActionableFeature] = [
    ActionableFeature("income_monthly_avg",        10_000, 100_000, True,  "Monthly income",       "₹"),
    ActionableFeature("income_consistency_score",  0.3, 1.0,         True,  "Income consistency"),
    ActionableFeature("income_regularity",         0.3, 1.0,         True,  "Income regularity"),
    ActionableFeature("expense_to_income_ratio",   0.1, 0.9,         False, "Expense-to-income ratio"),
    ActionableFeature("savings_potential",         0.0, 0.5,         True,  "Savings potential"),
    ActionableFeature("discipline_overall_score",  0.3, 1.0,         True,  "Financial discipline"),
    ActionableFeature("discipline_bill_timeliness", 0.4, 1.0,        True,  "Bill payment timeliness"),
    ActionableFeature("discipline_emi_regularity", 0.4, 1.0,         True,  "EMI payment regularity"),
    ActionableFeature("discretionary_spending_ratio", 0.0, 0.5,     False, "Discretionary spending"),
]

MAX_CHANGES = 3       # narrate at most this many adjustments
SWEEP_STEPS = 10      # values to test per feature
MIN_USEFUL_LIFT = 0.02  # ignore changes that move score < this much


def _predict_score(predictor: ProfitabilityPredictor, features: Dict[str, float]) -> float:
    df = pd.DataFrame([features])
    return float(predictor.predict(df, return_confidence=False))


def _best_change_for_feature(
    predictor: ProfitabilityPredictor,
    features: Dict[str, float],
    feat: ActionableFeature,
) -> Optional[Dict[str, Any]]:
    """Sweep one feature, return the single most score-lifting value (or None)."""
    if feat.name not in features:
        return None
    current = float(features[feat.name])

    # Build sweep values strictly *toward improvement* relative to current
    if feat.higher_is_better:
        target_top = max(current * 1.5, feat.realistic_max)
        candidates = np.linspace(max(current + 1e-6, feat.realistic_min), target_top, SWEEP_STEPS)
        candidates = candidates[candidates > current]
    else:
        target_bottom = min(current * 0.5, feat.realistic_min)
        candidates = np.linspace(min(current - 1e-6, feat.realistic_max), target_bottom, SWEEP_STEPS)
        candidates = candidates[candidates < current]

    if len(candidates) == 0:
        return None

    base_score = _predict_score(predictor, features)
    best: Optional[Dict[str, Any]] = None

    for val in candidates:
        trial = dict(features)
        trial[feat.name] = float(val)
        s = _predict_score(predictor, trial)
        lift = s - base_score
        if lift < MIN_USEFUL_LIFT:
            continue
        if best is None or lift > best["delta_score"]:
            best = {
                "feature": feat.name,
                "display_label": feat.display_label,
                "display_unit": feat.display_unit,
                "current": round(current, 4),
                "suggested": round(float(val), 4),
                "delta_score": round(float(lift), 4),
                "new_score": round(float(s), 4),
            }

    return best


def find_counterfactual(
    predictor: ProfitabilityPredictor,
    features: Dict[str, float],
    target_threshold: float = APPROVE_THRESHOLD,
    max_changes: int = MAX_CHANGES,
) -> Dict[str, Any]:
    """Greedy search for the minimum set of changes that pushes score over
    `target_threshold`.

    Returns
    -------
    {
      "reachable": bool,                # could we hit the threshold?
      "starting_score": float,
      "final_score": float,
      "changes": [
         {feature, display_label, display_unit, current, suggested,
          delta_score, new_score}, ...
      ]
    }
    """
    working = dict(features)
    start_score = _predict_score(predictor, working)
    changes: List[Dict[str, Any]] = []

    # Already above threshold — nothing to suggest
    if start_score >= target_threshold:
        return {
            "reachable": True,
            "starting_score": round(start_score, 4),
            "final_score": round(start_score, 4),
            "changes": [],
        }

    tried: set[str] = set()
    for _ in range(max_changes):
        # Score each unused actionable feature; greedy pick the biggest lift
        candidates: List[Dict[str, Any]] = []
        for af in ACTIONABLE_FEATURES:
            if af.name in tried:
                continue
            best = _best_change_for_feature(predictor, working, af)
            if best is not None:
                candidates.append(best)

        if not candidates:
            break

        candidates.sort(key=lambda c: c["delta_score"], reverse=True)
        chosen = candidates[0]
        changes.append(chosen)
        working[chosen["feature"]] = chosen["suggested"]
        tried.add(chosen["feature"])

        # Early exit if we've crossed the threshold
        if chosen["new_score"] >= target_threshold:
            break

    final_score = _predict_score(predictor, working)
    return {
        "reachable": final_score >= target_threshold,
        "starting_score": round(start_score, 4),
        "final_score": round(final_score, 4),
        "changes": changes,
    }
