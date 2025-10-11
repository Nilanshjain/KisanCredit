"""Behavioral Features (10% weight in profitability score).

Extracts red flag features from behavioral data.
"""

from typing import Dict


class BehavioralFeatureExtractor:
    """Extract behavioral red flag features."""

    def extract_features(self, behavioral_data: Dict) -> dict:
        """Extract all behavioral features.

        Args:
            behavioral_data: Dictionary with behavioral indicators

        Returns:
            Dictionary of behavioral features
        """
        gambling = behavioral_data.get('gambling_app_usage', False)
        location_changes = behavioral_data.get('frequent_location_changes', 0)
        night_ratio = behavioral_data.get('night_transaction_ratio', 0.0)
        fin_score = behavioral_data.get('financial_app_usage_score', 0.0)

        return {
            # Feature 1: Gambling app usage flag
            'behavioral_gambling_flag': 1 if gambling else 0,

            # Feature 2: Location change frequency
            'behavioral_location_changes': location_changes,

            # Feature 3: Location stability score
            'behavioral_location_stability': self._stability_score(location_changes),

            # Feature 4: Night transaction ratio
            'behavioral_night_transaction_ratio': night_ratio,

            # Feature 5: Financial app usage (digital literacy)
            'behavioral_financial_literacy': fin_score,

            # Feature 6: Overall behavioral risk score
            'behavioral_risk_score': self._risk_score(
                gambling, location_changes, night_ratio, fin_score
            ),
        }

    def _stability_score(self, location_changes: int) -> float:
        """Calculate location stability (higher = more stable)."""
        if location_changes >= 20:
            return 0.0
        elif location_changes >= 10:
            return 0.3
        elif location_changes >= 5:
            return 0.6
        else:
            return 1.0

    def _risk_score(
        self,
        gambling: bool,
        location_changes: int,
        night_ratio: float,
        fin_score: float
    ) -> float:
        """Calculate overall behavioral risk score (0-100, higher = riskier)."""
        risk = 0.0

        # Gambling adds 40 points
        if gambling:
            risk += 40

        # Location changes add up to 30 points
        if location_changes > 10:
            risk += 30
        elif location_changes > 5:
            risk += 15

        # Night transactions add up to 30 points
        if night_ratio > 0.3:
            risk += 30
        elif night_ratio > 0.2:
            risk += 20
        elif night_ratio > 0.1:
            risk += 10

        # Low financial literacy adds risk
        if fin_score < 3:
            risk += 20
        elif fin_score < 5:
            risk += 10

        return min(100.0, risk)
