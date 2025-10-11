"""Financial Discipline Features (10% weight in profitability score).

Extracts features from SMS transaction data to measure financial responsibility.
"""

import pandas as pd
from typing import List


class FinancialDisciplineExtractor:
    """Extract financial discipline features from SMS transactions."""

    def extract_features(self, sms_transactions: List[dict]) -> dict:
        """Extract all financial discipline features.

        Args:
            sms_transactions: List of SMS transaction dictionaries

        Returns:
            Dictionary of financial discipline features
        """
        if not sms_transactions:
            return self._get_zero_features()

        # Convert to DataFrame
        df = pd.DataFrame(sms_transactions)
        df['amount'] = df['amount'].astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        debits = df[df['transaction_type'] == 'debit'].copy()
        credits = df[df['transaction_type'] == 'credit'].copy()

        return {
            # Feature 1: Loan EMI payment regularity
            'discipline_emi_regularity': self._emi_regularity(debits),

            # Feature 2: Bill payment timeliness score
            'discipline_bill_timeliness': self._bill_timeliness(debits),

            # Feature 3: Failed transaction ratio (bounces)
            'discipline_failed_transaction_ratio': self._failed_ratio(df),

            # Feature 4: Savings behavior score
            'discipline_savings_behavior': self._savings_behavior(credits, debits),

            # Feature 5: Credit utilization pattern
            'discipline_credit_usage_score': self._credit_usage(df),

            # Feature 6: Overall financial discipline score (0-100)
            'discipline_overall_score': self._overall_discipline_score(
                self._emi_regularity(debits),
                self._bill_timeliness(debits),
                self._failed_ratio(df),
                self._savings_behavior(credits, debits)
            ),
        }

    def _emi_regularity(self, debits: pd.DataFrame) -> float:
        """Calculate loan EMI payment regularity.

        Regular EMI payments indicate financial discipline.
        """
        emi_transactions = debits[debits['source'] == 'Loan EMI']

        if len(emi_transactions) == 0:
            return 0.5  # Neutral score (no loans)

        # Calculate monthly frequency
        min_date = emi_transactions['timestamp'].min()
        max_date = emi_transactions['timestamp'].max()
        months = max((max_date - min_date).days / 30, 1)

        frequency = len(emi_transactions) / months

        # Expected: 1 EMI per month
        # Score: closer to 1.0 = better
        if 0.9 <= frequency <= 1.1:
            regularity = 1.0
        elif 0.7 <= frequency <= 1.3:
            regularity = 0.7
        elif frequency > 1.3:
            regularity = 0.5  # Multiple loans (risky)
        else:
            regularity = 0.3  # Missed payments

        return round(regularity, 4)

    def _bill_timeliness(self, debits: pd.DataFrame) -> float:
        """Calculate bill payment timeliness.

        Regular bills (electricity, mobile) paid on time indicate discipline.
        """
        bill_categories = ['Electricity Bill', 'Mobile Recharge', 'Insurance']
        bills = debits[debits['source'].isin(bill_categories)]

        if len(bills) == 0:
            return 0.0

        # Calculate payment frequency
        min_date = bills['timestamp'].min()
        max_date = bills['timestamp'].max()
        months = max((max_date - min_date).days / 30, 1)

        frequency = len(bills) / months

        # Expected: 2-3 bills per month (electricity + mobile)
        if frequency >= 2.0:
            timeliness = 1.0
        elif frequency >= 1.0:
            timeliness = 0.6
        else:
            timeliness = 0.3

        return round(timeliness, 4)

    def _failed_ratio(self, df: pd.DataFrame) -> float:
        """Calculate failed transaction ratio.

        High failed transaction rate indicates insufficient funds (risky).
        """
        # In real implementation, would check for keywords like "failed", "bounce"
        # For now, use a proxy: very small transactions followed by reversals
        total_transactions = len(df)

        if total_transactions == 0:
            return 0.0

        # Proxy: count transactions with "Failed" or "Bounce" in message
        failed = df[df['message'].str.contains('fail|bounce|reject', case=False, na=False)]

        failed_ratio = len(failed) / total_transactions
        return round(failed_ratio, 4)

    def _savings_behavior(self, credits: pd.DataFrame, debits: pd.DataFrame) -> float:
        """Calculate savings behavior score.

        Positive savings trend indicates financial planning.
        """
        if len(credits) == 0:
            return 0.0

        total_income = credits['amount'].sum()
        total_expense = debits['amount'].sum() if len(debits) > 0 else 0

        # Savings rate
        savings_rate = (total_income - total_expense) / total_income if total_income > 0 else 0

        # Score based on savings rate
        if savings_rate >= 0.3:
            savings_score = 1.0
        elif savings_rate >= 0.2:
            savings_score = 0.8
        elif savings_rate >= 0.1:
            savings_score = 0.6
        elif savings_rate >= 0:
            savings_score = 0.4
        else:
            savings_score = 0.0  # Spending more than earning

        return round(savings_score, 4)

    def _credit_usage(self, df: pd.DataFrame) -> float:
        """Calculate credit card usage pattern.

        Moderate credit usage with full payments = good.
        High credit usage = risky.
        """
        # Check for credit card transactions
        credit_txns = df[df['message'].str.contains('credit card|card payment', case=False, na=False)]

        if len(credit_txns) == 0:
            return 0.5  # Neutral (no credit card)

        # Count payments vs charges
        payments = credit_txns[credit_txns['transaction_type'] == 'debit']
        charges = credit_txns[credit_txns['transaction_type'] == 'credit']

        if len(charges) == 0:
            return 0.5

        # Healthy: payments >= charges (paying off credit)
        payment_ratio = len(payments) / len(charges)

        if payment_ratio >= 1.0:
            credit_score = 1.0
        elif payment_ratio >= 0.7:
            credit_score = 0.7
        else:
            credit_score = 0.4

        return round(credit_score, 4)

    def _overall_discipline_score(
        self,
        emi_regularity: float,
        bill_timeliness: float,
        failed_ratio: float,
        savings_behavior: float
    ) -> float:
        """Calculate overall financial discipline score (0-100).

        Weighted combination of all discipline metrics.
        """
        # Positive factors
        positive_score = (
            emi_regularity * 30 +  # EMI regularity (30%)
            bill_timeliness * 25 +  # Bill timeliness (25%)
            savings_behavior * 30   # Savings behavior (30%)
        )

        # Negative factors
        penalty = failed_ratio * 15  # Failed transactions (15% penalty)

        overall = max(0, positive_score - penalty)
        return round(overall, 2)

    def _get_zero_features(self) -> dict:
        """Return zero values when no transaction data."""
        return {
            'discipline_emi_regularity': 0.0,
            'discipline_bill_timeliness': 0.0,
            'discipline_failed_transaction_ratio': 0.0,
            'discipline_savings_behavior': 0.0,
            'discipline_credit_usage_score': 0.0,
            'discipline_overall_score': 0.0,
        }
