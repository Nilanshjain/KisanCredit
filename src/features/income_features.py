"""Income Stability Features (40% weight in profitability score).

Extracts features from SMS transaction data to measure income stability.
"""

import pandas as pd
import numpy as np
from typing import List
from decimal import Decimal


class IncomeFeatureExtractor:
    """Extract income stability features from SMS transactions."""

    def extract_features(self, sms_transactions: List[dict]) -> dict:
        """Extract all income-related features.

        Args:
            sms_transactions: List of SMS transaction dictionaries

        Returns:
            Dictionary of income features
        """
        # Convert to DataFrame for vectorized operations
        df = pd.DataFrame(sms_transactions)

        # Filter credit transactions (income)
        credits = df[df['transaction_type'] == 'credit'].copy()

        if len(credits) == 0:
            return self._get_zero_features()

        # Convert amount to float
        credits['amount'] = credits['amount'].astype(float)
        credits['timestamp'] = pd.to_datetime(credits['timestamp'])

        return {
            # Feature 1: Average monthly income
            'income_monthly_avg': self._monthly_avg(credits),

            # Feature 2: Income consistency (coefficient of variation)
            'income_consistency_score': self._consistency_score(credits),

            # Feature 3: Number of unique income sources
            'income_source_diversity': self._source_diversity(credits),

            # Feature 4: Frequency of transactions
            'income_transaction_frequency': self._transaction_frequency(credits),

            # Feature 5: Income growth trend
            'income_growth_trend': self._growth_trend(credits),

            # Feature 6: Regularity of income (std/mean)
            'income_regularity': self._regularity(credits),

            # Feature 7: Largest transaction amount
            'income_max_transaction': credits['amount'].max(),

            # Feature 8: Smallest transaction amount
            'income_min_transaction': credits['amount'].min(),

            # Feature 9: Income quartile ratio (Q3/Q1)
            'income_quartile_ratio': self._quartile_ratio(credits),
        }

    def _monthly_avg(self, credits: pd.DataFrame) -> float:
        """Calculate average monthly income."""
        if len(credits) == 0:
            return 0.0

        # Calculate months span
        min_date = credits['timestamp'].min()
        max_date = credits['timestamp'].max()
        months = max(1, (max_date - min_date).days / 30)

        total_income = credits['amount'].sum()
        return round(total_income / months, 2)

    def _consistency_score(self, credits: pd.DataFrame) -> float:
        """Coefficient of variation (lower = more consistent)."""
        if len(credits) < 2:
            return 0.0

        mean = credits['amount'].mean()
        std = credits['amount'].std()

        if mean == 0:
            return 0.0

        cv = std / mean
        # Convert to score (higher = better)
        consistency = max(0, 1 - cv)
        return round(consistency, 4)

    def _source_diversity(self, credits: pd.DataFrame) -> int:
        """Number of unique income sources."""
        return credits['source'].nunique()

    def _transaction_frequency(self, credits: pd.DataFrame) -> float:
        """Average transactions per month."""
        if len(credits) == 0:
            return 0.0

        min_date = credits['timestamp'].min()
        max_date = credits['timestamp'].max()
        months = max(1, (max_date - min_date).days / 30)

        return round(len(credits) / months, 2)

    def _growth_trend(self, credits: pd.DataFrame) -> float:
        """Income growth trend (slope of regression)."""
        if len(credits) < 2:
            return 0.0

        # Sort by date
        credits = credits.sort_values('timestamp')

        # Calculate days from start
        credits['days'] = (credits['timestamp'] - credits['timestamp'].min()).dt.days

        # Simple linear regression slope
        x = credits['days'].values
        y = credits['amount'].values

        if len(x) < 2:
            return 0.0

        slope = np.polyfit(x, y, 1)[0]
        return round(slope, 4)

    def _regularity(self, credits: pd.DataFrame) -> float:
        """Income regularity score."""
        if len(credits) < 2:
            return 0.0

        mean = credits['amount'].mean()
        std = credits['amount'].std()

        if mean == 0:
            return 0.0

        # Lower std/mean = more regular
        regularity = 1 / (1 + (std / mean))
        return round(regularity, 4)

    def _quartile_ratio(self, credits: pd.DataFrame) -> float:
        """Q3/Q1 ratio (income distribution)."""
        if len(credits) < 4:
            return 1.0

        q1 = credits['amount'].quantile(0.25)
        q3 = credits['amount'].quantile(0.75)

        if q1 == 0:
            return 1.0

        return round(q3 / q1, 4)

    def _get_zero_features(self) -> dict:
        """Return zero values when no income data."""
        return {
            'income_monthly_avg': 0.0,
            'income_consistency_score': 0.0,
            'income_source_diversity': 0,
            'income_transaction_frequency': 0.0,
            'income_growth_trend': 0.0,
            'income_regularity': 0.0,
            'income_max_transaction': 0.0,
            'income_min_transaction': 0.0,
            'income_quartile_ratio': 1.0,
        }
