"""Pydantic schemas for API request/response validation."""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class DecisionEnum(str, Enum):
    """Loan decision types."""
    approve = "approve"
    reject = "reject"
    manual_review = "manual_review"


class SMSTransaction(BaseModel):
    """SMS transaction data."""
    transaction_id: str
    timestamp: datetime
    amount: float
    transaction_type: str
    merchant_category: Optional[str] = None
    is_credit: bool


class ContactMetadata(BaseModel):
    """Contact metadata for social network analysis."""
    total_contacts: int = Field(ge=0)
    family_contacts: int = Field(ge=0)
    business_contacts: int = Field(ge=0)
    government_contacts: int = Field(ge=0)
    avg_call_duration: float = Field(ge=0)
    contact_diversity_score: float = Field(ge=0, le=1)


class LocationPattern(BaseModel):
    """Location pattern data."""
    unique_locations: int = Field(ge=0)
    home_location: Dict[str, float]  # {"lat": float, "lon": float}
    travel_radius_km: float = Field(ge=0)
    area_type: str  # "urban", "semi_urban", "rural"
    location_stability_score: float = Field(ge=0, le=1)


class BehavioralData(BaseModel):
    """Behavioral pattern data."""
    app_usage_hours_per_day: float = Field(ge=0, le=24)
    night_activity_ratio: float = Field(ge=0, le=1)
    gambling_indicators: int = Field(ge=0)
    financial_app_usage: bool
    literacy_score: float = Field(ge=0, le=1)


class LoanApplicationRequest(BaseModel):
    """Request schema for loan application submission."""
    user_id: str = Field(..., min_length=1, max_length=100)
    loan_amount: float = Field(..., gt=0, description="Requested loan amount in INR")
    loan_purpose: str = Field(..., min_length=1)

    # Alternative data
    sms_transactions: List[SMSTransaction] = Field(..., min_items=0, max_items=10000)
    contact_metadata: ContactMetadata
    location_pattern: LocationPattern
    behavioral_data: BehavioralData

    # Optional traditional data
    monthly_income: Optional[float] = Field(None, ge=0)
    employment_type: Optional[str] = None

    @validator('loan_amount')
    def validate_loan_amount(cls, v):
        """Validate loan amount is within acceptable range."""
        if v > 500000:  # Max 5 Lakh
            raise ValueError('Loan amount cannot exceed ₹5,00,000')
        if v < 1000:  # Min 1000
            raise ValueError('Loan amount must be at least ₹1,000')
        return v

    class Config:
        schema_extra = {
            "example": {
                "user_id": "USER123",
                "loan_amount": 50000.0,
                "loan_purpose": "Agricultural equipment",
                "sms_transactions": [
                    {
                        "transaction_id": "TXN001",
                        "timestamp": "2024-01-15T10:30:00",
                        "amount": 5000.0,
                        "transaction_type": "credit",
                        "merchant_category": "agriculture",
                        "is_credit": True
                    }
                ],
                "contact_metadata": {
                    "total_contacts": 150,
                    "family_contacts": 20,
                    "business_contacts": 50,
                    "government_contacts": 5,
                    "avg_call_duration": 180.0,
                    "contact_diversity_score": 0.75
                },
                "location_pattern": {
                    "unique_locations": 5,
                    "home_location": {"lat": 28.7041, "lon": 77.1025},
                    "travel_radius_km": 25.0,
                    "area_type": "rural",
                    "location_stability_score": 0.85
                },
                "behavioral_data": {
                    "app_usage_hours_per_day": 4.5,
                    "night_activity_ratio": 0.15,
                    "gambling_indicators": 0,
                    "financial_app_usage": True,
                    "literacy_score": 0.7
                }
            }
        }


class PredictionRequest(BaseModel):
    """Request schema for direct prediction (features already extracted)."""
    application_id: str
    features: Dict[str, float] = Field(..., description="Pre-extracted feature dictionary")

    class Config:
        schema_extra = {
            "example": {
                "application_id": "APP123",
                "features": {
                    "income_monthly_avg": 25000.0,
                    "expense_to_income_ratio": 0.65,
                    "social_network_strength": 0.75
                }
            }
        }


class FeatureContribution(BaseModel):
    """Feature contribution in prediction explanation."""
    feature: str
    value: float
    contribution: float
    importance: float


class PredictionResponse(BaseModel):
    """Response schema for prediction results."""
    application_id: str
    user_id: Optional[str] = None
    profitability_score: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    decision: DecisionEnum
    decision_threshold: float = 0.6
    prediction_timestamp: datetime
    processing_time_ms: float

    class Config:
        schema_extra = {
            "example": {
                "application_id": "APP123",
                "user_id": "USER123",
                "profitability_score": 0.78,
                "confidence": 0.85,
                "decision": "approve",
                "decision_threshold": 0.6,
                "prediction_timestamp": "2024-01-15T10:30:00",
                "processing_time_ms": 25.3
            }
        }


class ExplanationResponse(BaseModel):
    """Response schema for prediction explanation."""
    application_id: str
    profitability_score: float
    decision: DecisionEnum
    base_value: float
    top_contributors: List[FeatureContribution]
    explanation_timestamp: datetime

    class Config:
        schema_extra = {
            "example": {
                "application_id": "APP123",
                "profitability_score": 0.78,
                "decision": "approve",
                "base_value": 0.50,
                "top_contributors": [
                    {
                        "feature": "income_consistency_score",
                        "value": 0.85,
                        "contribution": 0.12,
                        "importance": 450.5
                    }
                ],
                "explanation_timestamp": "2024-01-15T10:30:00"
            }
        }


class ApplicationResponse(BaseModel):
    """Response schema for application submission."""
    application_id: str
    user_id: str
    status: str
    profitability_score: float
    decision: DecisionEnum
    submitted_at: datetime
    processing_time_ms: float
    message: str

    class Config:
        schema_extra = {
            "example": {
                "application_id": "APP123",
                "user_id": "USER123",
                "status": "processed",
                "profitability_score": 0.78,
                "decision": "approve",
                "submitted_at": "2024-01-15T10:30:00",
                "processing_time_ms": 45.8,
                "message": "Application processed successfully"
            }
        }


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    timestamp: datetime
    version: str
    model_loaded: bool
    model_health: Dict[str, Any]
    uptime_seconds: float

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00",
                "version": "1.0.0",
                "model_loaded": True,
                "model_health": {
                    "is_healthy": True,
                    "prediction_latency_ms": 8.5
                },
                "uptime_seconds": 3600.0
            }
        }


class MetricsResponse(BaseModel):
    """Metrics response schema."""
    total_predictions: int
    predictions_last_hour: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    approval_rate: float
    cache_hit_rate: float

    class Config:
        schema_extra = {
            "example": {
                "total_predictions": 15420,
                "predictions_last_hour": 345,
                "avg_latency_ms": 18.5,
                "p95_latency_ms": 35.2,
                "p99_latency_ms": 48.7,
                "error_rate": 0.002,
                "approval_rate": 0.68,
                "cache_hit_rate": 0.72
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    message: str
    status_code: int
    timestamp: datetime
    request_id: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid input data",
                "status_code": 400,
                "timestamp": "2024-01-15T10:30:00",
                "request_id": "req_123"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request schema."""
    applications: List[LoanApplicationRequest] = Field(..., min_items=1, max_items=100)

    @validator('applications')
    def validate_batch_size(cls, v):
        """Validate batch size."""
        if len(v) > 100:
            raise ValueError('Batch size cannot exceed 100 applications')
        return v


class BatchPredictionResponse(BaseModel):
    """Batch prediction response schema."""
    batch_id: str
    total_applications: int
    successful_predictions: int
    failed_predictions: int
    results: List[PredictionResponse]
    processing_time_ms: float
    timestamp: datetime
