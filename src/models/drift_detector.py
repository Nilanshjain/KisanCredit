"""Population Stability Index (PSI) drift detection.

PSI is the industry-standard tabular-data drift metric for credit-risk models:
it compares the distribution of a feature in a reference window (e.g. the
training set) against the distribution in a current window (recent
predictions). Values:

    PSI < 0.10   stable
    0.10-0.25    moderate drift — investigate
    PSI > 0.25   significant drift — model may need retraining

No external ML dependencies; pure numpy. ~30 lines of real logic.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np


def compute_psi(
    reference: Sequence[float],
    current: Sequence[float],
    n_bins: int = 10,
) -> float:
    """PSI between a reference distribution and a current one.

    Bins reference into `n_bins` quantiles (so each bin has ~equal mass in the
    reference), then assigns `current` into the same bins and compares the
    proportion in each bin to the reference proportion.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.size < n_bins or cur.size == 0:
        return 0.0

    # Bin edges from the reference quantiles. Add tiny epsilon to the last edge
    # so the right boundary is inclusive.
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(ref, quantiles)
    edges[0] -= 1e-9
    edges[-1] += 1e-9

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    # Convert to proportions; smooth zeros with epsilon to keep log() finite.
    eps = 1e-6
    ref_prop = (ref_counts / max(ref.size, 1)) + eps
    cur_prop = (cur_counts / max(cur.size, 1)) + eps

    psi = float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))
    return round(psi, 4)


def compute_psi_from_quantiles(
    reference_quantiles: Sequence[float],
    current_values: Sequence[float],
) -> float:
    """PSI when you only have the reference bin edges (not the raw reference)."""
    edges = np.asarray(reference_quantiles, dtype=float)
    cur = np.asarray(current_values, dtype=float)
    if edges.size < 2 or cur.size == 0:
        return 0.0

    edges = edges.copy()
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    n_bins = edges.size - 1

    cur_counts, _ = np.histogram(cur, bins=edges)
    eps = 1e-6
    # Each reference bin is by construction equal-mass = 1/n_bins.
    ref_prop = np.full(n_bins, 1.0 / n_bins) + eps
    cur_prop = (cur_counts / max(cur.size, 1)) + eps
    psi = float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))
    return round(psi, 4)


def severity(psi: float) -> str:
    """Map a PSI value to one of three severity bands."""
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "moderate"
    return "significant"


def compute_feature_drift(
    reference_quantiles: Optional[Dict[str, List[float]]],
    current_rows: List[Dict[str, float]],
    feature_names: List[str],
) -> List[Dict]:
    """Per-feature PSI summary for an admin dashboard.

    Returns a list of dicts:
        [{feature, psi, severity, n_current, baseline_available}, ...]

    Sorted descending by PSI so the noisiest features bubble up. Missing
    baseline -> psi=None and severity='unknown'.
    """
    if not current_rows:
        return [
            {
                "feature": f,
                "psi": None,
                "severity": "no_data",
                "n_current": 0,
                "baseline_available": bool(reference_quantiles and f in reference_quantiles),
            }
            for f in feature_names
        ]

    out: List[Dict] = []
    for feature in feature_names:
        current_values = [row[feature] for row in current_rows if feature in row]
        baseline_edges = (reference_quantiles or {}).get(feature)

        if not baseline_edges:
            out.append({
                "feature": feature,
                "psi": None,
                "severity": "unknown",
                "n_current": len(current_values),
                "baseline_available": False,
            })
            continue

        psi = compute_psi_from_quantiles(baseline_edges, current_values)
        out.append({
            "feature": feature,
            "psi": psi,
            "severity": severity(psi),
            "n_current": len(current_values),
            "baseline_available": True,
        })

    # Sort: unknown last, otherwise descending by PSI
    out.sort(key=lambda d: (d["psi"] is None, -(d["psi"] or 0)))
    return out
