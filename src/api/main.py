"""Main FastAPI application for KisanCredit profitability scoring API.

Provides endpoints for:
- Loan application submission and scoring
- Profitability predictions
- Model explanations
- Health checks and metrics
"""

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware as FastAPICORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict
import time
import uuid

from .schemas import (
    LoanApplicationRequest,
    PredictionRequest,
    PredictionResponse,
    ExplanationResponse,
    ApplicationResponse,
    HealthResponse,
    MetricsResponse,
    ErrorResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    DecisionEnum,
    FeatureContribution
)
from .middleware import (
    RateLimitMiddleware,
    RequestLoggingMiddleware,
    PerformanceMonitoringMiddleware
)

from ..models import ProfitabilityPredictor, ModelExplainer
from ..features import FeatureEngineeringPipeline
from ..utils.logger import get_logger
from ..utils.config import settings

logger = get_logger(__name__)

# Global instances
predictor: ProfitabilityPredictor = None
explainer: ModelExplainer = None
feature_pipeline: FeatureEngineeringPipeline = None
performance_monitor: PerformanceMonitoringMiddleware = None
app_start_time: float = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for application startup and shutdown."""
    # Startup
    global predictor, explainer, feature_pipeline, app_start_time

    logger.info("Starting KisanCredit API...")
    app_start_time = time.time()

    try:
        # Load model
        logger.info("Loading profitability model...")
        predictor = ProfitabilityPredictor(model_path=settings.model_path)

        # Initialize explainer
        if predictor.model:
            logger.info("Initializing SHAP explainer...")
            explainer = ModelExplainer(predictor.model, predictor.feature_names)
        else:
            logger.warning("Model not loaded. Explainer not initialized.")

        # Initialize feature pipeline
        logger.info("Initializing feature engineering pipeline...")
        feature_pipeline = FeatureEngineeringPipeline()

        logger.info("✓ KisanCredit API started successfully")

    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down KisanCredit API...")


# Create FastAPI app
app = FastAPI(
    title="KisanCredit API",
    description="AI-powered loan underwriting API for rural India",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS
app.add_middleware(
    FastAPICORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware, requests_limit=100, window_seconds=900)

# Add performance monitoring
performance_monitor = PerformanceMonitoringMiddleware(app)
app.add_middleware(PerformanceMonitoringMiddleware)


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.__class__.__name__,
            message=exc.detail,
            status_code=exc.status_code,
            timestamp=datetime.utcnow(),
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            status_code=500,
            timestamp=datetime.utcnow(),
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "KisanCredit API",
        "version": "1.0.0",
        "description": "AI-powered loan underwriting for rural India",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


# Health check endpoint
@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Check API and model health status."""
    model_health = predictor.health_check() if predictor else {"is_healthy": False}

    uptime = time.time() - app_start_time if app_start_time else 0

    return HealthResponse(
        status="healthy" if model_health.get("is_healthy") else "unhealthy",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        model_loaded=predictor is not None and predictor.model is not None,
        model_health=model_health,
        uptime_seconds=uptime
    )


# Metrics endpoint
@app.get("/api/v1/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get API performance metrics."""
    if not performance_monitor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Performance monitoring not available"
        )

    metrics = performance_monitor.get_metrics()

    # Calculate additional metrics
    total_requests = metrics['total_requests']
    predictions_last_hour = 0  # Would need time-series data

    # Approval rate (would come from stored predictions)
    approval_rate = 0.68  # Placeholder

    # Cache hit rate (would come from cache layer)
    cache_hit_rate = 0.0  # Placeholder (no cache yet)

    return MetricsResponse(
        total_predictions=total_requests,
        predictions_last_hour=predictions_last_hour,
        avg_latency_ms=metrics['avg_latency_ms'],
        p95_latency_ms=metrics['p95_latency_ms'],
        p99_latency_ms=metrics['p99_latency_ms'],
        error_rate=metrics['error_rate'],
        approval_rate=approval_rate,
        cache_hit_rate=cache_hit_rate
    )


# Application submission endpoint
@app.post("/api/v1/applications", response_model=ApplicationResponse, status_code=status.HTTP_201_CREATED)
async def submit_application(application: LoanApplicationRequest):
    """Submit loan application and get profitability decision.

    This endpoint:
    1. Receives raw application data
    2. Extracts features
    3. Makes profitability prediction
    4. Returns decision with score
    """
    start_time = time.time()

    if not predictor or not feature_pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not available. Model not loaded."
        )

    try:
        # Generate application ID
        application_id = f"APP_{uuid.uuid4().hex[:12].upper()}"

        logger.info(
            "Processing application",
            application_id=application_id,
            user_id=application.user_id,
            loan_amount=application.loan_amount
        )

        # Convert to dictionary format for feature extraction
        app_data = {
            'application_id': application_id,
            'user_id': application.user_id,
            'sms_transactions': [txn.dict() for txn in application.sms_transactions],
            'contact_metadata': application.contact_metadata.dict(),
            'location_pattern': application.location_pattern.dict(),
            'behavioral_data': application.behavioral_data.dict()
        }

        # Extract features
        features = feature_pipeline.extract_features(app_data)

        # Remove metadata
        import pandas as pd
        features_df = pd.DataFrame([features])
        metadata_cols = ['application_id', 'user_id']
        feature_cols = [col for col in features_df.columns if col not in metadata_cols]
        X = features_df[feature_cols]

        # Make prediction
        result = predictor.predict(X, return_confidence=True)

        processing_time = (time.time() - start_time) * 1000

        logger.info(
            "Application processed",
            application_id=application_id,
            score=result['score'],
            decision=result['decision'],
            processing_time_ms=processing_time
        )

        return ApplicationResponse(
            application_id=application_id,
            user_id=application.user_id,
            status="processed",
            profitability_score=result['score'],
            decision=DecisionEnum(result['decision']),
            submitted_at=datetime.utcnow(),
            processing_time_ms=processing_time,
            message="Application processed successfully"
        )

    except Exception as e:
        logger.error(f"Application processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Application processing failed: {str(e)}"
        )


# Direct prediction endpoint
@app.post("/api/v1/predictions", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make profitability prediction from pre-extracted features.

    Use this endpoint when features are already extracted.
    """
    start_time = time.time()

    if not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not available. Model not loaded."
        )

    try:
        import pandas as pd

        # Convert features to DataFrame
        features_df = pd.DataFrame([request.features])

        # Make prediction
        result = predictor.predict(features_df, return_confidence=True)

        processing_time = (time.time() - start_time) * 1000

        return PredictionResponse(
            application_id=request.application_id,
            user_id=None,
            profitability_score=result['score'],
            confidence=result['confidence'],
            decision=DecisionEnum(result['decision']),
            decision_threshold=0.6,
            prediction_timestamp=datetime.utcnow(),
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


# Explanation endpoint
@app.get("/api/v1/predictions/{application_id}/explain", response_model=ExplanationResponse)
async def explain_prediction(application_id: str, features: Dict[str, float]):
    """Get SHAP-based explanation for a prediction.

    Requires the same features used for prediction.
    """
    if not predictor or not explainer:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Explainer not available. Model or explainer not loaded."
        )

    try:
        import pandas as pd

        # Convert features to DataFrame
        features_df = pd.DataFrame([features])

        # Get explanation
        explanation = explainer.explain_prediction(features_df, top_n=10)

        # Convert to response format
        top_contributors = [
            FeatureContribution(
                feature=contrib['feature'],
                value=contrib['value'],
                contribution=contrib['shap_value'],
                importance=contrib['abs_shap_value']
            )
            for contrib in explanation['top_contributions']
        ]

        return ExplanationResponse(
            application_id=application_id,
            profitability_score=explanation['prediction'],
            decision=DecisionEnum(explanation['decision']),
            base_value=explanation['base_value'],
            top_contributors=top_contributors,
            explanation_timestamp=datetime.utcnow()
        )

    except Exception as e:
        logger.error(f"Explanation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation failed: {str(e)}"
        )


# Batch prediction endpoint
@app.post("/api/v1/predictions/batch", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """Process batch of loan applications.

    Maximum 100 applications per batch.
    """
    start_time = time.time()

    if not predictor or not feature_pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not available."
        )

    try:
        batch_id = f"BATCH_{uuid.uuid4().hex[:12].upper()}"
        results = []
        failed_count = 0

        logger.info(
            "Processing batch",
            batch_id=batch_id,
            n_applications=len(request.applications)
        )

        # Process each application
        for app_request in request.applications:
            try:
                # Generate application ID
                application_id = f"APP_{uuid.uuid4().hex[:12].upper()}"

                # Convert to dict for feature extraction
                app_data = {
                    'application_id': application_id,
                    'user_id': app_request.user_id,
                    'sms_transactions': [txn.dict() for txn in app_request.sms_transactions],
                    'contact_metadata': app_request.contact_metadata.dict(),
                    'location_pattern': app_request.location_pattern.dict(),
                    'behavioral_data': app_request.behavioral_data.dict()
                }

                # Extract features and predict
                features = feature_pipeline.extract_features(app_data)

                import pandas as pd
                features_df = pd.DataFrame([features])
                metadata_cols = ['application_id', 'user_id']
                feature_cols = [col for col in features_df.columns if col not in metadata_cols]
                X = features_df[feature_cols]

                result = predictor.predict(X, return_confidence=True)

                results.append(PredictionResponse(
                    application_id=application_id,
                    user_id=app_request.user_id,
                    profitability_score=result['score'],
                    confidence=result['confidence'],
                    decision=DecisionEnum(result['decision']),
                    decision_threshold=0.6,
                    prediction_timestamp=datetime.utcnow(),
                    processing_time_ms=result['prediction_time_ms']
                ))

            except Exception as e:
                logger.error(f"Failed to process application in batch: {e}")
                failed_count += 1

        processing_time = (time.time() - start_time) * 1000

        logger.info(
            "Batch processing completed",
            batch_id=batch_id,
            successful=len(results),
            failed=failed_count,
            processing_time_ms=processing_time
        )

        return BatchPredictionResponse(
            batch_id=batch_id,
            total_applications=len(request.applications),
            successful_predictions=len(results),
            failed_predictions=failed_count,
            results=results,
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        logger.error(f"Batch processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info"
    )
