"""Synthetic data generator for loan applications.

Generates realistic 100K+ loan applications with:
- SMS transaction data (income/expenses)
- Contact metadata (social network)
- Location patterns (geospatial stability)
- Behavioral indicators (red flags)
"""

import random
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any
import numpy as np
import pandas as pd

from src.pipeline.schemas import (
    SMSTransaction,
    ContactMetadata,
    LocationPattern,
    BehavioralData,
    LoanApplication
)


class SyntheticDataGenerator:
    """Generate realistic synthetic loan application data."""

    def __init__(self, seed: int = 42):
        """Initialize generator with random seed."""
        random.seed(seed)
        np.random.seed(seed)

        # SMS transaction templates
        self.income_sources = [
            "UPI", "Bank Transfer", "Mandi Sale", "Customer Payment",
            "Government Subsidy", "Rent Income"
        ]
        self.expense_categories = [
            "Electricity Bill", "Mobile Recharge", "Grocery", "Medical",
            "Fuel", "School Fees", "Loan EMI", "Insurance"
        ]

        # Location coordinates (rural India)
        self.rural_locations = [
            (28.6139, 77.2090),  # Delhi NCR
            (19.0760, 72.8777),  # Mumbai rural
            (13.0827, 80.2707),  # Chennai rural
            (22.5726, 88.3639),  # Kolkata rural
            (12.9716, 77.5946),  # Bangalore rural
        ]

    def generate_sms_transactions(
        self,
        monthly_income: Decimal,
        num_months: int = 6
    ) -> List[SMSTransaction]:
        """Generate realistic SMS transaction history."""
        transactions = []
        base_date = datetime.now() - timedelta(days=30 * num_months)

        # Generate income transactions (40% weight in profitability score)
        num_income_txns = random.randint(15, 40)
        for i in range(num_income_txns):
            days_offset = random.randint(0, 30 * num_months)
            amount = float(monthly_income) * random.uniform(0.2, 0.4)

            source = random.choice(self.income_sources)
            message = f"Credited Rs.{amount:.2f} to A/c from {source}"

            transactions.append(SMSTransaction(
                timestamp=base_date + timedelta(days=days_offset),
                transaction_type="credit",
                amount=Decimal(str(round(amount, 2))),
                source=source,
                message=message,
                category="income"
            ))

        # Generate expense transactions (25% weight)
        num_expense_txns = random.randint(20, 60)
        total_expenses = float(monthly_income) * random.uniform(0.6, 0.9) * num_months

        for i in range(num_expense_txns):
            days_offset = random.randint(0, 30 * num_months)
            amount = total_expenses / num_expense_txns * random.uniform(0.5, 1.5)

            category = random.choice(self.expense_categories)
            message = f"Debited Rs.{amount:.2f} for {category}"

            transactions.append(SMSTransaction(
                timestamp=base_date + timedelta(days=days_offset),
                transaction_type="debit",
                amount=Decimal(str(round(amount, 2))),
                source=category,
                message=message,
                category="expense"
            ))

        # Sort by timestamp
        transactions.sort(key=lambda x: x.timestamp)
        return transactions[:100]  # Limit to 100 transactions

    def generate_contact_metadata(self, social_strength: str) -> ContactMetadata:
        """Generate contact metadata based on social network strength.

        Args:
            social_strength: 'strong', 'medium', 'weak'
        """
        if social_strength == "strong":
            total = random.randint(200, 500)
            family = random.randint(20, 50)
            business = random.randint(30, 80)
            government = random.randint(2, 10)
            comm_freq = random.uniform(30, 80)
        elif social_strength == "medium":
            total = random.randint(100, 200)
            family = random.randint(10, 25)
            business = random.randint(15, 40)
            government = random.randint(0, 5)
            comm_freq = random.uniform(15, 35)
        else:  # weak
            total = random.randint(50, 100)
            family = random.randint(5, 15)
            business = random.randint(5, 20)
            government = random.randint(0, 2)
            comm_freq = random.uniform(5, 20)

        return ContactMetadata(
            total_contacts=total,
            family_contacts=family,
            business_contacts=business,
            government_contacts=government,
            avg_communication_frequency=round(comm_freq, 2)
        )

    def generate_location_pattern(self, location_type: str) -> LocationPattern:
        """Generate location pattern data."""
        base_lat, base_lon = random.choice(self.rural_locations)

        # Add some randomness around base location
        lat = base_lat + random.uniform(-0.5, 0.5)
        lon = base_lon + random.uniform(-0.5, 0.5)

        if location_type == "rural":
            stability = random.uniform(0.7, 1.0)
            distance = random.uniform(50, 200)
            travel_freq = random.randint(0, 5)
        elif location_type == "semi_urban":
            stability = random.uniform(0.5, 0.8)
            distance = random.uniform(20, 80)
            travel_freq = random.randint(3, 10)
        else:  # urban
            stability = random.uniform(0.3, 0.6)
            distance = random.uniform(0, 30)
            travel_freq = random.randint(8, 20)

        return LocationPattern(
            latitude=round(lat, 6),
            longitude=round(lon, 6),
            location_type=location_type,
            stability_score=round(stability, 2),
            distance_from_financial_center=round(distance, 2),
            travel_frequency=travel_freq
        )

    def generate_behavioral_data(self, risk_level: str) -> BehavioralData:
        """Generate behavioral indicators.

        Args:
            risk_level: 'low', 'medium', 'high'
        """
        if risk_level == "low":
            gambling = False
            location_changes = random.randint(0, 3)
            night_ratio = random.uniform(0.0, 0.1)
            financial_score = random.uniform(6, 10)
        elif risk_level == "medium":
            gambling = random.choice([True, False])
            location_changes = random.randint(3, 10)
            night_ratio = random.uniform(0.1, 0.3)
            financial_score = random.uniform(4, 7)
        else:  # high risk
            gambling = True
            location_changes = random.randint(10, 30)
            night_ratio = random.uniform(0.3, 0.6)
            financial_score = random.uniform(1, 5)

        return BehavioralData(
            gambling_app_usage=gambling,
            frequent_location_changes=location_changes,
            night_transaction_ratio=round(night_ratio, 2),
            financial_app_usage_score=round(financial_score, 1)
        )

    def calculate_profitability_score(
        self,
        monthly_income: float,
        sms_transactions: List[SMSTransaction],
        contact_metadata: ContactMetadata,
        behavioral_data: BehavioralData,
        location_pattern: LocationPattern
    ) -> float:
        """Calculate profitability score using PDF weights.

        Weights from PDF:
        - Income Stability: 40%
        - Expense Management: 25%
        - Social Network: 15%
        - Financial Discipline: 10%
        - Behavioral Red Flags: 10%
        """
        # Income stability (40%)
        credits = [t for t in sms_transactions if t.transaction_type == "credit"]
        income_consistency = 1.0 if len(credits) > 10 else len(credits) / 10
        income_score = income_consistency * 100

        # Expense management (25%)
        debits = [t for t in sms_transactions if t.transaction_type == "debit"]
        total_credits = sum(float(t.amount) for t in credits)
        total_debits = sum(float(t.amount) for t in debits)
        expense_ratio = total_debits / total_credits if total_credits > 0 else 1.0
        expense_score = max(0, (1 - expense_ratio)) * 100

        # Social network strength (15%)
        social_score = min(100, (contact_metadata.total_contacts / 500) * 100)

        # Financial discipline (10%)
        discipline_score = behavioral_data.financial_app_usage_score * 10

        # Behavioral red flags (10%)
        behavioral_score = 100
        if behavioral_data.gambling_app_usage:
            behavioral_score -= 40
        if behavioral_data.frequent_location_changes > 10:
            behavioral_score -= 30
        if behavioral_data.night_transaction_ratio > 0.3:
            behavioral_score -= 30
        behavioral_score = max(0, behavioral_score)

        # Weighted sum
        profitability = (
            income_score * 0.40 +
            expense_score * 0.25 +
            social_score * 0.15 +
            discipline_score * 0.10 +
            behavioral_score * 0.10
        )

        return round(profitability, 2)

    def generate_application(self) -> LoanApplication:
        """Generate a single loan application."""
        # Random profile
        age = random.randint(25, 60)
        gender = random.choice(["male", "female", "other"])
        occupation = random.choice([
            "farmer", "shopkeeper", "daily_wage", "self_employed", "salaried"
        ])

        # Income based on occupation
        income_ranges = {
            "farmer": (8000, 25000),
            "shopkeeper": (15000, 40000),
            "daily_wage": (6000, 15000),
            "self_employed": (20000, 60000),
            "salaried": (25000, 80000)
        }
        monthly_income = Decimal(str(random.randint(*income_ranges[occupation])))

        # Loan amount (1-12 months of income)
        loan_amount = monthly_income * Decimal(str(random.randint(1, 12)))

        # Determine profiles for data generation
        social_strength = random.choice(["strong", "medium", "weak"])
        location_type = random.choice(["rural", "semi_urban", "urban"])
        risk_level = random.choice(["low", "medium", "high"])

        # Generate alternative data
        sms_transactions = self.generate_sms_transactions(monthly_income)
        contact_metadata = self.generate_contact_metadata(social_strength)
        location_pattern = self.generate_location_pattern(location_type)
        behavioral_data = self.generate_behavioral_data(risk_level)

        # Calculate profitability score
        profitability_score = self.calculate_profitability_score(
            float(monthly_income),
            sms_transactions,
            contact_metadata,
            behavioral_data,
            location_pattern
        )

        # Determine approval and default based on profitability
        approved = profitability_score >= 50
        default_status = profitability_score < 40 if approved else None

        return LoanApplication(
            application_id=str(uuid.uuid4()),
            user_id=f"user_{uuid.uuid4().hex[:8]}",
            age=age,
            gender=gender,
            occupation=occupation,
            monthly_income=monthly_income,
            loan_amount_requested=loan_amount,
            loan_purpose=random.choice([
                "Agricultural equipment", "Business expansion",
                "Medical emergency", "Education", "Home renovation"
            ]),
            sms_transactions=sms_transactions,
            contact_metadata=contact_metadata,
            location_pattern=location_pattern,
            behavioral_data=behavioral_data,
            approved=approved,
            default_status=default_status,
            profitability_score=profitability_score
        )

    def generate_dataset(self, num_applications: int = 100000) -> pd.DataFrame:
        """Generate complete dataset of loan applications.

        Args:
            num_applications: Number of applications to generate (default 100K)

        Returns:
            pandas DataFrame with all applications
        """
        print(f"Generating {num_applications} synthetic loan applications...")

        applications = []
        for i in range(num_applications):
            if i % 10000 == 0:
                print(f"Progress: {i}/{num_applications} applications generated")

            try:
                app = self.generate_application()
                applications.append(app.dict())
            except Exception as e:
                print(f"Error generating application {i}: {e}")
                continue

        print(f"[OK] Generated {len(applications)} valid applications")

        # Convert to DataFrame
        df = pd.DataFrame(applications)
        return df

    def save_dataset(
        self,
        df: pd.DataFrame,
        output_path: str = "data/synthetic/loan_applications.parquet"
    ) -> None:
        """Save dataset to parquet format."""
        df.to_parquet(output_path, index=False, compression='snappy')
        print(f"[OK] Dataset saved to {output_path}")
        print(f"  Size: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
