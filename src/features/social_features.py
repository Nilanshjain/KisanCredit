"""Social Network Features (15% weight in profitability score).

Extracts features from contact metadata to measure social capital.
"""

from typing import Dict


class SocialFeatureExtractor:
    """Extract social network features from contact metadata."""

    def extract_features(self, contact_metadata: Dict) -> dict:
        """Extract all social network features.

        Args:
            contact_metadata: Dictionary with contact information

        Returns:
            Dictionary of social network features
        """
        total = contact_metadata.get('total_contacts', 0)
        family = contact_metadata.get('family_contacts', 0)
        business = contact_metadata.get('business_contacts', 0)
        government = contact_metadata.get('government_contacts', 0)
        comm_freq = contact_metadata.get('avg_communication_frequency', 0)

        return {
            # Feature 1: Total contacts (normalized)
            'social_total_contacts': total,

            # Feature 2: Network strength score (0-100)
            'social_network_strength': self._network_strength(total),

            # Feature 3: Family network size
            'social_family_size': family,

            # Feature 4: Business network size
            'social_business_size': business,

            # Feature 5: Government contact presence
            'social_government_connections': government,

            # Feature 6: Communication frequency
            'social_communication_frequency': comm_freq,

            # Feature 7: Network diversity score
            'social_network_diversity': self._network_diversity(
                family, business, government, total
            ),

            # Feature 8: Business to total ratio
            'social_business_ratio': self._business_ratio(business, total),
        }

    def _network_strength(self, total: int) -> float:
        """Calculate network strength score (0-100)."""
        # Strong network: 200+ contacts
        # Weak network: <100 contacts
        if total >= 500:
            return 100.0
        elif total >= 200:
            return 60.0 + (total - 200) / 300 * 40  # Scale 60-100
        elif total >= 100:
            return 30.0 + (total - 100) / 100 * 30  # Scale 30-60
        else:
            return total / 100 * 30  # Scale 0-30

    def _network_diversity(
        self,
        family: int,
        business: int,
        government: int,
        total: int
    ) -> float:
        """Calculate network diversity using entropy."""
        if total == 0:
            return 0.0

        # Calculate proportions
        other = max(0, total - family - business - government)
        proportions = [family, business, government, other]
        proportions = [p / total for p in proportions if p > 0]

        # Shannon entropy
        import math
        entropy = -sum(p * math.log(p + 1e-10) for p in proportions)

        # Normalize to 0-1
        max_entropy = math.log(4)  # max for 4 categories
        diversity = entropy / max_entropy if max_entropy > 0 else 0

        return round(diversity, 4)

    def _business_ratio(self, business: int, total: int) -> float:
        """Business contacts ratio."""
        if total == 0:
            return 0.0
        return round(business / total, 4)
