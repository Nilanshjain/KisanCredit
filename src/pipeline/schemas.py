"""Pydantic schemas for data validation."""

from datetime import datetime
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, validator
from decimal import Decimal


class SMSTransaction(BaseModel):
    """SMS transaction data from bank/UPI alerts."""

    timestamp: datetime
    transaction_type: Literal["credit", "debit"]
    amount: Decimal = Field(gt=0, description="Transaction amount")
    source: str = Field(description="Bank/UPI/Mandi/Utility")
    message: str = Field(max_length=500, description="SMS content")
    category: Optional[str] = Field(default=None, description="Transaction category")


class ContactMetadata(BaseModel):
    """Contact list metadata for social network analysis."""

    total_contacts: int = Field(ge=0, le=1000, description="Total contacts")
    family_contacts: int = Field(ge=0, description="Family members")
    business_contacts: int = Field(ge=0, description="Business contacts")
    government_contacts: int = Field(ge=0, description="Authority contacts")
    avg_communication_frequency: float = Field(ge=0, le=100, description="Calls per month")


class LocationPattern(BaseModel):
    """Geolocation pattern data."""

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    location_type: Literal["rural", "semi_urban", "urban"]
    stability_score: float = Field(ge=0, le=1, description="0=nomadic, 1=stable")
    distance_from_financial_center: float = Field(ge=0, description="Distance in km")
    travel_frequency: int = Field(ge=0, le=30, description="Days traveled per month")


class BehavioralData(BaseModel):
    """Behavioral indicators and red flags."""

    gambling_app_usage: bool = Field(default=False)
    frequent_location_changes: int = Field(ge=0, le=100, description="Location changes per month")
    night_transaction_ratio: float = Field(ge=0, le=1, description="Ratio of night transactions")
    financial_app_usage_score: float = Field(ge=0, le=10, description="Digital literacy indicator")


class LoanApplication(BaseModel):
    """Complete loan application with all data sources."""

    application_id: str = Field(description="Unique application ID")
    user_id: str = Field(description="User identifier")

    # Basic info
    age: int = Field(ge=18, le=80)
    gender: Literal["male", "female", "other"]
    occupation: Literal["farmer", "shopkeeper", "daily_wage", "self_employed", "salaried"]
    monthly_income: Decimal = Field(gt=0, description="Declared monthly income")
    loan_amount_requested: Decimal = Field(gt=0, description="Requested loan amount")
    loan_purpose: str = Field(max_length=200)

    # Alternative data
    sms_transactions: List[SMSTransaction] = Field(min_items=10, max_items=200)
    contact_metadata: ContactMetadata
    location_pattern: LocationPattern
    behavioral_data: BehavioralData

    # Ground truth (for training)
    approved: Optional[bool] = None
    default_status: Optional[bool] = None
    profitability_score: Optional[float] = Field(default=None, ge=0, le=100)

    created_at: datetime = Field(default_factory=datetime.now)

    @validator('loan_amount_requested')
    def validate_loan_amount(cls, v, values):
        """Validate loan amount is reasonable compared to income."""
        if 'monthly_income' in values:
            if v > values['monthly_income'] * 24:  # Max 2 years of income
                raise ValueError("Loan amount too high compared to income")
        return v


class DataQualityMetrics(BaseModel):
    """Data quality metrics for monitoring."""

    total_records: int
    valid_records: int
    invalid_records: int
    null_counts: dict
    validation_errors: List[str]
    processing_time_seconds: float
    timestamp: datetime = Field(default_factory=datetime.now)

    @property
    def quality_score(self) -> float:
        """Calculate overall quality score."""
        if self.total_records == 0:
            return 0.0
        return (self.valid_records / self.total_records) * 100
