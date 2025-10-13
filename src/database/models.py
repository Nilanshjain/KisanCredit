"""SQLAlchemy database models for KisanCredit.

Models:
- User: User accounts and profiles
- Application: Loan applications
- Prediction: Model predictions and decisions
- AuditLog: Audit trail for compliance
"""

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, JSON, ForeignKey, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid

Base = declarative_base()


def generate_uuid():
    """Generate UUID for primary keys."""
    return str(uuid.uuid4())


class User(Base):
    """User model for loan applicants."""

    __tablename__ = "users"

    # Primary key
    id = Column(String(50), primary_key=True, default=generate_uuid)

    # Basic information
    user_id = Column(String(100), unique=True, nullable=False, index=True)
    phone_number = Column(String(20), unique=True, index=True)
    email = Column(String(255), unique=True, nullable=True, index=True)
    full_name = Column(String(255))

    # KYC information
    aadhaar_number = Column(String(12), unique=True, nullable=True)
    pan_number = Column(String(10), unique=True, nullable=True)
    kyc_verified = Column(Boolean, default=False)
    kyc_verified_at = Column(DateTime, nullable=True)

    # Profile data
    date_of_birth = Column(DateTime, nullable=True)
    address = Column(Text, nullable=True)
    city = Column(String(100), nullable=True)
    state = Column(String(100), nullable=True)
    pincode = Column(String(10), nullable=True, index=True)

    # Employment
    employment_type = Column(String(50), nullable=True)  # farmer, self-employed, salaried
    monthly_income = Column(Float, nullable=True)

    # Account status
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    applications = relationship("Application", back_populates="user", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index('idx_user_phone', 'phone_number'),
        Index('idx_user_email', 'email'),
        Index('idx_user_active', 'is_active'),
        Index('idx_user_created', 'created_at'),
    )

    def __repr__(self):
        return f"<User(id={self.id}, user_id={self.user_id}, phone={self.phone_number})>"


class Application(Base):
    """Loan application model."""

    __tablename__ = "applications"

    # Primary key
    id = Column(String(50), primary_key=True, default=generate_uuid)

    # Foreign key
    user_id = Column(String(50), ForeignKey("users.id"), nullable=False, index=True)

    # Application details
    application_id = Column(String(100), unique=True, nullable=False, index=True)
    loan_amount = Column(Float, nullable=False)
    loan_purpose = Column(String(255), nullable=False)
    loan_tenure_months = Column(Integer, nullable=True)  # Tenure in months

    # Application status
    status = Column(
        String(50),
        default="submitted",
        nullable=False,
        index=True
    )  # submitted, processing, approved, rejected, disbursed

    # Alternative data (stored as JSON)
    sms_transactions = Column(JSON, nullable=True)
    contact_metadata = Column(JSON, nullable=True)
    location_pattern = Column(JSON, nullable=True)
    behavioral_data = Column(JSON, nullable=True)

    # Extracted features (stored as JSON for audit)
    extracted_features = Column(JSON, nullable=True)

    # Timestamps
    submitted_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    processed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Processing metadata
    processing_time_ms = Column(Float, nullable=True)

    # Relationships
    user = relationship("User", back_populates="applications")
    predictions = relationship("Prediction", back_populates="application", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="application", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index('idx_app_user', 'user_id'),
        Index('idx_app_status', 'status'),
        Index('idx_app_submitted', 'submitted_at'),
        Index('idx_app_amount', 'loan_amount'),
    )

    def __repr__(self):
        return f"<Application(id={self.id}, application_id={self.application_id}, status={self.status})>"


class Prediction(Base):
    """Model prediction results."""

    __tablename__ = "predictions"

    # Primary key
    id = Column(String(50), primary_key=True, default=generate_uuid)

    # Foreign key
    application_id = Column(String(50), ForeignKey("applications.id"), nullable=False, index=True)

    # Prediction details
    prediction_id = Column(String(100), unique=True, nullable=False, index=True)
    profitability_score = Column(Float, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    decision = Column(String(50), nullable=False, index=True)  # approve, reject, manual_review
    decision_threshold = Column(Float, default=0.6)

    # Model information
    model_version = Column(String(50), nullable=True)
    model_name = Column(String(100), default="profitability_model")

    # Feature contributions (SHAP values as JSON)
    feature_contributions = Column(JSON, nullable=True)
    top_features = Column(JSON, nullable=True)  # Top contributing features

    # Prediction metadata
    prediction_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    prediction_latency_ms = Column(Float, nullable=True)

    # Business metrics
    estimated_profitability_inr = Column(Float, nullable=True)
    risk_category = Column(String(50), nullable=True)  # low, medium, high

    # Relationships
    application = relationship("Application", back_populates="predictions")

    # Indexes
    __table_args__ = (
        Index('idx_pred_app', 'application_id'),
        Index('idx_pred_score', 'profitability_score'),
        Index('idx_pred_decision', 'decision'),
        Index('idx_pred_timestamp', 'prediction_timestamp'),
    )

    def __repr__(self):
        return f"<Prediction(id={self.id}, score={self.profitability_score}, decision={self.decision})>"


class AuditLog(Base):
    """Audit log for compliance and tracking."""

    __tablename__ = "audit_logs"

    # Primary key
    id = Column(String(50), primary_key=True, default=generate_uuid)

    # Foreign key (optional - for application-specific logs)
    application_id = Column(String(50), ForeignKey("applications.id"), nullable=True, index=True)

    # Event information
    event_type = Column(String(100), nullable=False, index=True)  # user_created, application_submitted, etc.
    event_category = Column(String(50), nullable=False, index=True)  # user, application, prediction, system
    event_description = Column(Text, nullable=True)

    # Actor information
    actor_id = Column(String(100), nullable=True)  # User or system identifier
    actor_type = Column(String(50), nullable=True)  # user, system, admin

    # Event data (stored as JSON)
    event_data = Column(JSON, nullable=True)
    event_metadata = Column(JSON, nullable=True)

    # IP and request information
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(String(500), nullable=True)
    request_id = Column(String(100), nullable=True, index=True)

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    application = relationship("Application", back_populates="audit_logs")

    # Indexes
    __table_args__ = (
        Index('idx_audit_type', 'event_type'),
        Index('idx_audit_category', 'event_category'),
        Index('idx_audit_created', 'created_at'),
        Index('idx_audit_request', 'request_id'),
    )

    def __repr__(self):
        return f"<AuditLog(id={self.id}, event_type={self.event_type}, created_at={self.created_at})>"


class ModelMetrics(Base):
    """Model performance metrics tracking."""

    __tablename__ = "model_metrics"

    # Primary key
    id = Column(String(50), primary_key=True, default=generate_uuid)

    # Time period
    period_start = Column(DateTime, nullable=False, index=True)
    period_end = Column(DateTime, nullable=False, index=True)
    period_type = Column(String(20), default="hourly")  # hourly, daily, weekly

    # Prediction metrics
    total_predictions = Column(Integer, default=0)
    total_approvals = Column(Integer, default=0)
    total_rejections = Column(Integer, default=0)
    approval_rate = Column(Float, nullable=True)

    # Performance metrics
    avg_profitability_score = Column(Float, nullable=True)
    avg_confidence = Column(Float, nullable=True)
    avg_latency_ms = Column(Float, nullable=True)
    p95_latency_ms = Column(Float, nullable=True)
    p99_latency_ms = Column(Float, nullable=True)

    # Error tracking
    total_errors = Column(Integer, default=0)
    error_rate = Column(Float, nullable=True)

    # Business metrics
    total_loan_amount = Column(Float, nullable=True)
    avg_loan_amount = Column(Float, nullable=True)

    # Additional metrics (JSON)
    detailed_metrics = Column(JSON, nullable=True)

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Indexes
    __table_args__ = (
        Index('idx_metrics_period_start', 'period_start'),
        Index('idx_metrics_period_end', 'period_end'),
        Index('idx_metrics_type', 'period_type'),
    )

    def __repr__(self):
        return f"<ModelMetrics(period={self.period_start}, predictions={self.total_predictions})>"


class CacheMetrics(Base):
    """Cache performance metrics."""

    __tablename__ = "cache_metrics"

    # Primary key
    id = Column(String(50), primary_key=True, default=generate_uuid)

    # Time tracking
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    period_minutes = Column(Integer, default=60)  # Aggregation period

    # Cache statistics
    total_requests = Column(Integer, default=0)
    cache_hits = Column(Integer, default=0)
    cache_misses = Column(Integer, default=0)
    hit_rate = Column(Float, nullable=True)

    # Performance
    avg_cache_latency_ms = Column(Float, nullable=True)
    avg_db_latency_ms = Column(Float, nullable=True)

    # Cache size
    total_keys = Column(Integer, nullable=True)
    memory_usage_mb = Column(Float, nullable=True)

    # Additional metrics
    evictions = Column(Integer, default=0)
    expirations = Column(Integer, default=0)

    # Indexes
    __table_args__ = (
        Index('idx_cache_timestamp', 'timestamp'),
    )

    def __repr__(self):
        return f"<CacheMetrics(timestamp={self.timestamp}, hit_rate={self.hit_rate})>"
