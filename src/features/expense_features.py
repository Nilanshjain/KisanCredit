"""Expense Management Features (25% weight in profitability score).

Extracts features from SMS transaction data to measure expense patterns.
"""

import pandas as pd
import numpy as np
from typing import List


class ExpenseFeatureExtractor:
    """Extract expense management features from SMS transactions."""

    def extract_features(self, sms_transactions: List[dict]) -> dict:
        """Extract all expense-related features.

        Args:
            sms_transactions: List of SMS transaction dictionaries

        Returns:
            Dictionary of expense features
        """
        # Convert to DataFrame
        df = pd.DataFrame(sms_transactions)

        # Filter debit transactions (expenses)
        debits = df[df['transaction_type'] == 'debit'].copy()
        credits = df[df['transaction_type'] == 'credit'].copy()

        if len(debits) == 0:
            return self._get_zero_features()

        # Convert amount to float
        debits['amount'] = debits['amount'].astype(float)
        debits['timestamp'] = pd.to_datetime(debits['timestamp'])

        if len(credits) > 0:
            credits['amount'] = credits['amount'].astype(float)

        return {
            # Feature 1: Average monthly expenses
            'expense_monthly_avg': self._monthly_avg(debits),

            # Feature 2: Expense to income ratio (DTI)
            'expense_to_income_ratio': self._dti_ratio(credits, debits),

            # Feature 3: Essential vs non-essential ratio
            'essential_expense_ratio': self._essential_ratio(debits),

            # Feature 4: Expense consistency
            'expense_consistency': self._consistency(debits),

            # Feature 5: Bill payment regularity
            'bill_payment_regularity': self._bill_regularity(debits),

            # Feature 6: Savings potential (income - expenses)
            'savings_potential': self._savings_potential(credits, debits),

            # Feature 7: Expense spike count
            'expense_spike_count': self._spike_count(debits),

            # Feature 8: Discretionary spending ratio
            'discretionary_spending_ratio': self._discretionary_ratio(debits),

            # Feature 9: Expense volatility
            'expense_volatility': self._volatility(debits),
        }

    def _monthly_avg(self, debits: pd.DataFrame) -> float:
        """Calculate average monthly expenses."""
        if len(debits) == 0:
            return 0.0

        min_date = debits['timestamp'].min()
        max_date = debits['timestamp'].max()
        months = max(1, (max_date - min_date).days / 30)

        total_expense = debits['amount'].sum()
        return round(total_expense / months, 2)

    def _dti_ratio(self, credits: pd.DataFrame, debits: pd.DataFrame) -> float:
        """Debt-to-income ratio."""
        if len(credits) == 0 or len(debits) == 0:
            return 0.0

        total_income = credits['amount'].sum()
        total_expense = debits['amount'].sum()

        if total_income == 0:
            return 1.0

        return round(total_expense / total_income, 4)

    def _essential_ratio(self, debits: pd.DataFrame) -> float:
        """Ratio of essential expenses (bills, medical, etc.)."""
        essential_categories = [
            'Electricity Bill', 'Medical', 'School Fees',
            'Insurance', 'Loan EMI'
        ]

        essential = debits[debits['source'].isin(essential_categories)]

        if len(debits) == 0:
            return 0.0

        return round(essential['amount'].sum() / debits['amount'].sum(), 4)

    def _consistency(self, debits: pd.DataFrame) -> float:
        """Expense consistency score."""
        if len(debits) < 2:
            return 0.0

        mean = debits['amount'].mean()
        std = debits['amount'].std()

        if mean == 0:
            return 0.0

        cv = std / mean
        consistency = max(0, 1 - cv)
        return round(consistency, 4)

    def _bill_regularity(self, debits: pd.DataFrame) -> float:
        """Bill payment regularity score."""
        bill_categories = ['Electricity Bill', 'Mobile Recharge']
        bills = debits[debits['source'].isin(bill_categories)]

        if len(bills) == 0:
            return 0.0

        # Calculate frequency
        min_date = bills['timestamp'].min()
        max_date = bills['timestamp'].max()
        months = max(1, (max_date - min_date).days / 30)

        frequency = len(bills) / months

        # Score based on monthly frequency (expect ~2-3 bills/month)
        regularity = min(1.0, frequency / 2.5)
        return round(regularity, 4)

    def _savings_potential(
        self,
        credits: pd.DataFrame,
        debits: pd.DataFrame
    ) -> float:
        """Savings potential (income - expenses)."""
        if len(credits) == 0:
            return 0.0

        total_income = credits['amount'].sum() if len(credits) > 0 else 0
        total_expense = debits['amount'].sum() if len(debits) > 0 else 0

        return round(total_income - total_expense, 2)

    def _spike_count(self, debits: pd.DataFrame) -> int:
        """Count of unusual expense spikes."""
        if len(debits) < 2:
            return 0

        mean = debits['amount'].mean()
        std = debits['amount'].std()

        # Spike = amount > mean + 2*std
        threshold = mean + 2 * std
        spikes = debits[debits['amount'] > threshold]

        return len(spikes)

    def _discretionary_ratio(self, debits: pd.DataFrame) -> float:
        """Discretionary (non-essential) spending ratio."""
        essential_categories = [
            'Electricity Bill', 'Medical', 'School Fees',
            'Insurance', 'Loan EMI', 'Grocery'
        ]

        discretionary = debits[~debits['source'].isin(essential_categories)]

        if len(debits) == 0:
            return 0.0

        return round(discretionary['amount'].sum() / debits['amount'].sum(), 4)

    def _volatility(self, debits: pd.DataFrame) -> float:
        """Expense volatility (standard deviation)."""
        if len(debits) < 2:
            return 0.0

        return round(debits['amount'].std(), 2)

    def _get_zero_features(self) -> dict:
        """Return zero values when no expense data."""
        return {
            'expense_monthly_avg': 0.0,
            'expense_to_income_ratio': 0.0,
            'essential_expense_ratio': 0.0,
            'expense_consistency': 0.0,
            'bill_payment_regularity': 0.0,
            'savings_potential': 0.0,
            'expense_spike_count': 0,
            'discretionary_spending_ratio': 0.0,
            'expense_volatility': 0.0,
        }
