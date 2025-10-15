"""FastAPI application module for KisanCredit API.

Provides REST API endpoints for:
- Loan application submission and profitability scoring
- Direct predictions from features
- SHAP-based explanations
- Batch predictions
- Health checks and metrics

Features:
- Rate limiting (100 req/15min)
- Structured logging
- Request validation
- Error handling
- Performance monitoring
- OpenAPI documentation
"""

from .main import app
from .schemas import (
    LoanApplicationRequest,
    PredictionRequest,
    PredictionResponse,
    ExplanationResponse,
    ApplicationResponse,
    HealthResponse,
    MetricsResponse
)

__all__ = [
    'app',
    'LoanApplicationRequest',
    'PredictionRequest',
    'PredictionResponse',
    'ExplanationResponse',
    'ApplicationResponse',
    'HealthResponse',
    'MetricsResponse'
]
