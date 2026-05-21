"""Pydantic schemas for API request/response validation."""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


__all__ = [
    "DecisionEnum",
    "SMSTransaction",
    "ContactMetadata",
    "LocationPattern",
    "BehavioralData",
    "LoanApplicationRequest",
    "PredictionRequest",
    "PredictionResponse",
    "ExplanationResponse",
    "NaturalLanguageExplanationModel",
    "CounterfactualChange",
    "CounterfactualResponse",
    "ApplicationResponse",
    "ApplicationSubmittedResponse",
    "ApplicationTimelineEvent",
    "ApplicationTimelineResponse",
    "ApplicationDetailResponse",
    "PredictionDetail",
    "HealthResponse",
    "MetricsResponse",
    "ErrorResponse",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "FeatureContribution",
    "SimpleLoanApplicationRequest",
    "SignupRequest",
    "LoginRequest",
]


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
    # Legacy columns the feature extractors read (named for the original
    # synthetic training set). Both default from sibling fields when omitted.
    source: Optional[str] = None
    message: Optional[str] = None
    is_credit: bool

    @validator('source', always=True)
    def _default_source_from_category(cls, v, values):
        return v or values.get('merchant_category')

    @validator('message', always=True)
    def _default_message(cls, v, values):
        if v is not None:
            return v
        # Build a plausible SMS body so the discipline extractor's keyword scans
        # (fail/bounce/reject, credit card) match nothing — i.e. clean record.
        kind = "credited" if values.get('is_credit') else "debited"
        source = values.get('source') or values.get('merchant_category') or 'merchant'
        amount = values.get('amount') or 0
        return f"Rs {amount:.0f} {kind} via {source}"


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


class NaturalLanguageExplanationModel(BaseModel):
    """LLM-narrated explanation accompanying the SHAP technical breakdown."""
    text: str
    suggestion: str = ""
    language: str = "en"
    source: str = "template"  # 'gemini' | 'template'
    cached: bool = False


class ExplanationResponse(BaseModel):
    """Response schema for prediction explanation."""
    application_id: str
    profitability_score: float
    decision: DecisionEnum
    base_value: float
    top_contributors: List[FeatureContribution]
    explanation_timestamp: datetime
    # Phase 6: Gemini-narrated plain-language explanation (always populated;
    # falls back to a deterministic template when LLM is unavailable).
    natural_language: Optional[NaturalLanguageExplanationModel] = None


class CounterfactualChange(BaseModel):
    feature: str
    display_label: str
    display_unit: str = ""
    current: float
    suggested: float
    delta_score: float
    new_score: float


class CounterfactualResponse(BaseModel):
    """How-to-improve suggestion derived from a greedy 1-D feature search."""
    application_id: str
    starting_score: float
    final_score: float
    reachable: bool                                  # could we cross the approve threshold?
    changes: List[CounterfactualChange]
    natural_language: Optional[NaturalLanguageExplanationModel] = None

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


class ApplicationSubmittedResponse(BaseModel):
    """Returned by POST /applications/simple in the async-lifecycle flow.

    No decision yet — caller should poll GET /applications/{id}/timeline (and
    the application detail endpoint) to watch the application progress through
    submitted -> under_review -> decided.
    """
    application_id: str
    status: str = "submitted"
    submitted_at: datetime
    message: str = "Application submitted. Processing will complete within ~15 seconds."

    class Config:
        schema_extra = {
            "example": {
                "application_id": "APP_ABC123456789",
                "status": "submitted",
                "submitted_at": "2026-05-20T17:00:00Z",
                "message": "Application submitted. Processing will complete within ~15 seconds.",
            }
        }


class ApplicationTimelineEvent(BaseModel):
    """A single status transition in an application's lifecycle."""
    from_status: Optional[str] = None
    to_status: str
    actor_type: str  # system | user | admin
    actor_id: Optional[str] = None
    reason: Optional[str] = None
    occurred_at: datetime


class ApplicationTimelineResponse(BaseModel):
    """Ordered list of status transitions for an application."""
    application_id: str
    current_status: str
    events: List[ApplicationTimelineEvent]

    class Config:
        schema_extra = {
            "example": {
                "application_id": "APP_ABC123456789",
                "current_status": "decided",
                "events": [
                    {"from_status": None, "to_status": "submitted", "actor_type": "user", "occurred_at": "2026-05-20T17:00:00Z"},
                    {"from_status": "submitted", "to_status": "under_review", "actor_type": "system", "occurred_at": "2026-05-20T17:00:05Z"},
                    {"from_status": "under_review", "to_status": "decided", "actor_type": "system", "occurred_at": "2026-05-20T17:00:15Z"},
                ],
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
    timestamp: str
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


class SimpleLoanApplicationRequest(BaseModel):
    """Simplified loan application request for frontend form submissions.

    This schema accepts basic personal and financial information from the frontend form
    and generates synthetic feature values for ML prediction in the demo.
    """
    # Personal Information
    name: str = Field(..., min_length=2, max_length=100, description="Applicant's full name")
    mobile: str = Field(..., pattern=r"^\d{10}$", description="10-digit mobile number")
    date_of_birth: str = Field(..., description="Date of birth in YYYY-MM-DD format")
    gender: str = Field(..., pattern=r"^(male|female|other)$", description="Gender")
    pincode: str = Field(..., pattern=r"^\d{6}$", description="6-digit pincode")
    occupation: str = Field(..., min_length=2, max_length=50, description="Occupation type")

    # Loan Information
    loan_amount: float = Field(..., gt=0, description="Requested loan amount in INR")
    loan_purpose: str = Field(..., min_length=2, max_length=100, description="Purpose of loan")

    # Financial Information
    monthly_income: float = Field(..., gt=0, description="Monthly income in INR")
    monthly_expenses: float = Field(..., gt=0, description="Monthly expenses in INR")

    @validator('loan_amount')
    def validate_loan_amount(cls, v):
        """Validate loan amount is within acceptable range."""
        if v > 500000:  # Max 5 Lakh
            raise ValueError('Loan amount cannot exceed ₹5,00,000')
        if v < 1000:  # Min 1000
            raise ValueError('Loan amount must be at least ₹1,000')
        return v

    @validator('monthly_expenses')
    def validate_expenses(cls, v, values):
        """Validate expenses are reasonable relative to income."""
        if 'monthly_income' in values and v > values['monthly_income'] * 1.5:
            raise ValueError('Monthly expenses cannot exceed 150% of monthly income')
        return v

    class Config:
        schema_extra = {
            "example": {
                "name": "Rajesh Kumar",
                "mobile": "9876543210",
                "date_of_birth": "1985-01-15",
                "gender": "male",
                "pincode": "110001",
                "occupation": "Farmer",
                "loan_amount": 50000,
                "loan_purpose": "Agriculture/Farming",
                "monthly_income": 35000,
                "monthly_expenses": 20000
            }
        }


class PredictionDetail(BaseModel):
    """Prediction detail for application history."""
    profitability_score: float
    decision: DecisionEnum
    confidence: float
    prediction_timestamp: datetime
    model_version: Optional[str] = None
    prediction_latency_ms: Optional[float] = None


class ApplicationDetailResponse(BaseModel):
    """Detailed application response with full history."""
    # Basic info
    application_id: str
    user_id: str
    status: str
    loan_amount: float
    loan_purpose: str
    submitted_at: datetime
    processed_at: Optional[datetime] = None
    processing_time_ms: Optional[float] = None

    # Full data
    sms_transactions: Optional[List[Dict[str, Any]]] = None
    contact_metadata: Optional[Dict[str, Any]] = None
    location_pattern: Optional[Dict[str, Any]] = None
    behavioral_data: Optional[Dict[str, Any]] = None
    extracted_features: Optional[Dict[str, float]] = None

    # Predictions
    predictions: List[PredictionDetail]

    class Config:
        schema_extra = {
            "example": {
                "application_id": "APP_ABC123456789",
                "user_id": "user_123",
                "status": "processed",
                "loan_amount": 50000,
                "loan_purpose": "General",
                "submitted_at": "2025-10-25T12:00:00Z",
                "processed_at": "2025-10-25T12:00:05Z",
                "processing_time_ms": 5234.5,
                "predictions": [
                    {
                        "profitability_score": 0.75,
                        "decision": "approve",
                        "confidence": 0.87,
                        "prediction_timestamp": "2025-10-25T12:00:05Z",
                        "model_version": "1.0"
                    }
                ]
            }
        }


class SignupRequest(BaseModel):
    """Request schema for user signup with email and password."""
    email: str = Field(..., min_length=3, max_length=255, description="User email address")
    password: str = Field(..., min_length=8, max_length=100, description="User password (min 8 characters)")
    full_name: str = Field(..., min_length=2, max_length=255, description="User's full name")
    phone_number: str = Field(..., pattern=r"^[6-9]\d{9}$", description="10-digit Indian mobile number")

    @validator('email')
    def validate_email(cls, v):
        """Validate email format."""
        import re
        if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', v):
            raise ValueError('Invalid email format')
        return v.lower()

    @validator('password')
    def validate_password(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v

    class Config:
        schema_extra = {
            "example": {
                "email": "rajesh.kumar@example.com",
                "password": "SecurePass123",
                "full_name": "Rajesh Kumar",
                "phone_number": "9876543210"
            }
        }


class LoginRequest(BaseModel):
    """Request schema for user login with email and password."""
    email: str = Field(..., min_length=3, max_length=255, description="User email address")
    password: str = Field(..., min_length=1, description="User password")

    @validator('email')
    def validate_email(cls, v):
        """Validate email format."""
        import re
        if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', v):
            raise ValueError('Invalid email format')
        return v.lower()

    class Config:
        schema_extra = {
            "example": {
                "email": "rajesh.kumar@example.com",
                "password": "SecurePass123"
            }
        }
