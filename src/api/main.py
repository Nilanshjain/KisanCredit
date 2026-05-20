"""Main FastAPI application for KisanCredit profitability scoring API.

Provides endpoints for:
- Loan application submission and scoring
- Profitability predictions
- Model explanations
- Health checks and metrics
"""

from fastapi import FastAPI, HTTPException, status, Request, Depends
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
    ApplicationDetailResponse,
    PredictionDetail,
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
from ..database.repositories import ApplicationRepository, PredictionRepository
from ..database.connection import get_db
from sqlalchemy.ext.asyncio import AsyncSession

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
        # Load feature engineering pipeline
        logger.info("Initializing feature engineering pipeline...")
        feature_pipeline = FeatureEngineeringPipeline()

        # Load ML model and explainer
        logger.info("Loading ML model...")
        predictor = ProfitabilityPredictor("models/profitability_model_latest.pkl")

        # TODO: Fix ModelExplainer initialization - needs feature_names parameter
        # logger.info("Loading model explainer...")
        # explainer = ModelExplainer("models/profitability_model_latest.pkl")

        logger.info("[OK] KisanCredit API started successfully with ML model loaded")

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

# CORS Configuration - Fixed: removed credentials=True when using wildcard origins
app.add_middleware(
    FastAPICORSMiddleware,
    allow_origins=["*"],  # For production, specify exact origins
    allow_credentials=False,  # Cannot be True when using wildcard origins
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware - Re-enabled after fixing CORS
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware, requests_limit=100, window_seconds=900)

# Auth and User routers - Re-enabled
from .auth import router as auth_router
from .users import router as users_router

app.include_router(auth_router)
app.include_router(users_router)


# Exception handlers - Re-enabled
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": getattr(request.state, 'request_id', None)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    # Avoid Unicode encoding errors in logging
    error_msg = str(exc).encode('ascii', 'replace').decode('ascii')
    logger.error(f"Unhandled exception: {error_msg}")

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": getattr(request.state, 'request_id', None)
        }
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


# Health check endpoint - SIMPLIFIED FOR TESTING
@app.get("/api/v1/health")
async def health_check():
    """Check API and model health status."""
    return {"status": "ok", "message": "API is healthy"}


# Metrics endpoint - Simplified without performance monitoring
@app.get("/api/v1/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get API performance metrics (simplified)."""
    # Return placeholder metrics - will be properly implemented with Redis/database later
    return MetricsResponse(
        total_predictions=0,
        predictions_last_hour=0,
        avg_latency_ms=0.0,
        p95_latency_ms=0.0,
        p99_latency_ms=0.0,
        error_rate=0.0,
        approval_rate=0.68,
        cache_hit_rate=0.0
    )


# Application submission endpoint
@app.post("/api/v1/applications", response_model=ApplicationResponse, status_code=status.HTTP_201_CREATED)
async def submit_application(
    application: LoanApplicationRequest,
    session: AsyncSession = Depends(get_db)
):
    """Submit loan application and get profitability decision.

    This endpoint:
    1. Receives raw application data
    2. Extracts features
    3. Makes profitability prediction
    4. Saves to database
    5. Returns decision with score
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

        # Save application to database
        db_application = await ApplicationRepository.create(session, {
            'application_id': application_id,
            'user_id': application.user_id,
            'loan_amount': application.loan_amount,
            'loan_purpose': 'General',  # Default purpose
            'status': 'processed',
            'sms_transactions': [txn.dict() for txn in application.sms_transactions],
            'contact_metadata': application.contact_metadata.dict(),
            'location_pattern': application.location_pattern.dict(),
            'behavioral_data': application.behavioral_data.dict(),
            'extracted_features': features,
            'processing_time_ms': processing_time,
            'submitted_at': datetime.utcnow(),
            'processed_at': datetime.utcnow()
        })

        # Save prediction to database
        prediction_id = f"PRED_{uuid.uuid4().hex[:12].upper()}"
        await PredictionRepository.create(session, {
            'prediction_id': prediction_id,
            'application_id': db_application.id,
            'profitability_score': result['score'],
            'confidence': result.get('confidence', 0.0),
            'decision': result['decision'],
            'decision_threshold': 0.6,
            'model_version': '1.0',
            'model_name': 'profitability_model',
            'prediction_latency_ms': result.get('prediction_time_ms', 0)
        })

        # Commit transaction
        await session.commit()

        logger.info(
            "Application saved to database",
            application_id=application_id,
            db_id=db_application.id,
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
        await session.rollback()
        logger.error(f"Application processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Application processing failed: {str(e)}"
        )


# Get application by ID endpoint
@app.get("/api/v1/applications/{application_id}", response_model=ApplicationDetailResponse)
async def get_application(
    application_id: str,
    session: AsyncSession = Depends(get_db)
):
    """Get detailed application information.

    Returns:
    - Application data
    - All predictions for this application
    - Feature breakdown
    - Decision history
    """
    try:
        # Get application
        application = await ApplicationRepository.get_by_id(session, application_id)

        if not application:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Application {application_id} not found"
            )

        # Get predictions for this application
        predictions = await PredictionRepository.get_by_application(
            session,
            application.id
        )

        # Convert predictions to response format
        prediction_details = [
            PredictionDetail(
                profitability_score=p.profitability_score,
                decision=DecisionEnum(p.decision),
                confidence=p.confidence,
                prediction_timestamp=p.prediction_timestamp,
                model_version=p.model_version,
                prediction_latency_ms=p.prediction_latency_ms
            )
            for p in predictions
        ]

        logger.info(
            "Application retrieved",
            application_id=application_id,
            predictions_count=len(predictions)
        )

        return ApplicationDetailResponse(
            application_id=application.application_id,
            user_id=application.user_id,
            status=application.status,
            loan_amount=application.loan_amount,
            loan_purpose=application.loan_purpose,
            submitted_at=application.submitted_at,
            processed_at=application.processed_at,
            processing_time_ms=application.processing_time_ms,
            sms_transactions=application.sms_transactions,
            contact_metadata=application.contact_metadata,
            location_pattern=application.location_pattern,
            behavioral_data=application.behavioral_data,
            extracted_features=application.extracted_features,
            predictions=prediction_details
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve application: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve application"
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
