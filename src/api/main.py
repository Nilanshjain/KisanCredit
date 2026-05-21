"""Main FastAPI application for KisanCredit profitability scoring API.

Provides endpoints for:
- Loan application submission and scoring
- Profitability predictions
- Model explanations
- Health checks and metrics
"""

# Force stdout/stderr to UTF-8 before any other module loads. On Windows the
# default codec is cp1252 which can't encode common chars (rupee symbol, bullet
# points, Unicode in tracebacks); structlog/uvicorn would crash mid-request
# with a UnicodeEncodeError. Linux/Render is UTF-8 by default so this is a
# no-op there. Must come before any "import" that might log on module load.
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import asyncio

from fastapi import FastAPI, HTTPException, status, Request, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware as FastAPICORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict
import time
import uuid

from .schemas import (
    LoanApplicationRequest,
    SimpleLoanApplicationRequest,
    PredictionRequest,
    PredictionResponse,
    ExplanationResponse,
    NaturalLanguageExplanationModel,
    CounterfactualChange,
    CounterfactualResponse,
    ApplicationResponse,
    ApplicationSubmittedResponse,
    ApplicationTimelineEvent,
    ApplicationTimelineResponse,
    ApplicationDetailResponse,
    PredictionDetail,
    HealthResponse,
    MetricsResponse,
    ErrorResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    DecisionEnum,
    FeatureContribution,
)
from .middleware import (
    RateLimitMiddleware,
    RequestLoggingMiddleware,
    PerformanceMonitoringMiddleware
)

from ..models import ProfitabilityPredictor, ModelExplainer
from ..features import FeatureEngineeringPipeline
from ..features.feature_engineering import FeatureExtractionError
from ..utils.logger import get_logger
from ..utils.config import settings
from ..database.repositories import (
    ApplicationRepository,
    PredictionRepository,
    ApplicationStatusRepository,
    UserRepository,
)
from ..cache import recent_features as recent_features_cache
from ..llm.gemini_explainer import explain_decision as llm_explain_decision, narrate_counterfactual as llm_narrate_counterfactual
from ..models.counterfactual import find_counterfactual
from ..database.connection import get_db, db_manager
from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger(__name__)

API_VERSION = "1.0.0"

# MODEL_VERSION is assigned at startup from the loaded artifact's metadata
# (see lifespan handler). Defaults to the legacy synthetic label until then.
MODEL_VERSION = "v1-synthetic"

# Global instances
predictor: ProfitabilityPredictor = None
explainer: ModelExplainer = None
feature_pipeline: FeatureEngineeringPipeline = None
app_start_time: float = None


def _derive_model_version(p: "ProfitabilityPredictor | None") -> str:
    """Human-readable version string derived from the loaded artifact's metadata."""
    if not p or not p.metadata:
        return "v1-synthetic"
    ds = p.metadata.get("dataset")
    if ds == "home-credit-default-risk":
        auc = p.metadata.get("lightgbm_cv_auc_mean")
        return f"v2-home-credit-auc={auc}" if auc else "v2-home-credit"
    return "v1-synthetic"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for application startup and shutdown."""
    global predictor, explainer, feature_pipeline, app_start_time

    logger.info("Starting KisanCredit API...")
    app_start_time = time.time()

    # Feature pipeline — pure compute, no I/O
    logger.info("Initializing feature engineering pipeline...")
    feature_pipeline = FeatureEngineeringPipeline()

    # ML model
    logger.info("Loading ML model...")
    predictor = ProfitabilityPredictor(settings.model_path)
    if predictor.model is None:
        raise RuntimeError(
            f"Model failed to load from {settings.model_path}. "
            "Cannot start without a model — fix MODEL_PATH or train one first."
        )

    # SHAP explainer — wires onto the already-loaded model + its feature names
    logger.info("Initializing SHAP explainer...")
    explainer = ModelExplainer(predictor.model, predictor.feature_names)

    # Derive model version from the artifact metadata (v1-synthetic vs v2-home-credit)
    global MODEL_VERSION
    MODEL_VERSION = _derive_model_version(predictor)

    # Database — connect once at startup; pool is reused per-request via get_db
    logger.info("Connecting to database...")
    try:
        await db_manager.connect()
    except Exception as e:
        # Don't crash the whole API if DB is briefly unreachable in dev/cold-start;
        # individual endpoints that need it will surface a clear 503.
        # Production deploys should monitor /health to catch persistent failures.
        logger.error(f"Database connection failed at startup (continuing): {e}")

    logger.info(
        "KisanCredit API started",
        api_version=API_VERSION,
        model_version=MODEL_VERSION,
        n_features=len(predictor.feature_names) if predictor.feature_names else 0,
    )

    yield

    logger.info("Shutting down KisanCredit API...")
    try:
        await db_manager.disconnect()
    except Exception as e:
        logger.warning(f"Database disconnect raised on shutdown: {e}")


# Create FastAPI app
app = FastAPI(
    title="KisanCredit API",
    description="AI-powered loan underwriting API for rural India",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS — origins from CORS_ORIGINS env var (whitelist required in production)
_cors_origins = settings.cors_origins_list()
app.add_middleware(
    FastAPICORSMiddleware,
    allow_origins=_cors_origins,
    # credentials=True is incompatible with wildcard origins; enable only if a real whitelist is set
    allow_credentials=_cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(PerformanceMonitoringMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware, requests_limit=100, window_seconds=900)

# Auth and User routers - Re-enabled
from .auth import router as auth_router
from .users import router as users_router
from .admin import router as admin_router

app.include_router(auth_router)
app.include_router(users_router)
app.include_router(admin_router)


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


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Liveness + readiness check.

    Returns 200 with status='healthy' only if the model is loaded and the DB
    answers a SELECT 1 within timeout. Returns 200 with status='degraded' if
    the API is up but a dependency is unreachable, so load balancers can
    distinguish "API is dead" (no response) from "API has a sick dependency"
    (status field tells the story).
    """
    model_health = predictor.health_check() if predictor else {"is_healthy": False, "model_loaded": False}
    db_health = await db_manager.health_check()

    overall_healthy = model_health.get("is_healthy") and db_health.get("is_healthy")
    uptime = time.time() - app_start_time if app_start_time else 0.0

    return HealthResponse(
        status="healthy" if overall_healthy else "degraded",
        timestamp=datetime.utcnow(),
        version=API_VERSION,
        model_loaded=bool(model_health.get("model_loaded")),
        model_health={
            **model_health,
            "model_version": MODEL_VERSION,
            "database": db_health,
        },
        uptime_seconds=round(uptime, 2),
    )


@app.get("/api/v1/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Real metrics from the PerformanceMonitoringMiddleware snapshot.

    Approval rate is derived from the prediction counter — see Phase 3 for
    bringing in per-decision counters. For now it reflects only request-level
    behaviour, not model-decision distribution.
    """
    snap = PerformanceMonitoringMiddleware.snapshot()
    return MetricsResponse(
        total_predictions=snap['total_requests'],
        predictions_last_hour=snap['requests_last_hour'],
        avg_latency_ms=snap['avg_latency_ms'],
        p95_latency_ms=snap['p95_latency_ms'],
        p99_latency_ms=snap['p99_latency_ms'],
        error_rate=round(snap['error_rate'], 4),
        approval_rate=0.0,  # populated once decision-level counters land in Phase 3
        cache_hit_rate=0.0,  # populated once Redis cache metrics are wired
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

        # model_dump(mode='json') ensures nested datetimes (in SMS timestamps) become ISO strings
        # so the JSON column serializer doesn't reject the payload.
        sms_dump = [txn.model_dump(mode='json') for txn in application.sms_transactions]
        contact_dump = application.contact_metadata.model_dump(mode='json')
        location_dump = application.location_pattern.model_dump(mode='json')
        behavioral_dump = application.behavioral_data.model_dump(mode='json')

        # Feature extractors take the raw dict shape — pydantic .model_dump() with mode='json'
        # gives strings for datetimes, which the extractors handle (they parse if needed).
        app_data = {
            'application_id': application_id,
            'user_id': application.user_id,
            'sms_transactions': sms_dump,
            'contact_metadata': contact_dump,
            'location_pattern': location_dump,
            'behavioral_data': behavioral_dump,
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

        db_application = await ApplicationRepository.create(session, {
            'application_id': application_id,
            'user_id': application.user_id,
            'loan_amount': application.loan_amount,
            'loan_purpose': 'General',
            'status': 'processed',
            'sms_transactions': sms_dump,
            'contact_metadata': contact_dump,
            'location_pattern': location_dump,
            'behavioral_data': behavioral_dump,
            'extracted_features': features,
            'processing_time_ms': processing_time,
            'submitted_at': datetime.utcnow(),
            'processed_at': datetime.utcnow(),
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

    except FeatureExtractionError as e:
        await session.rollback()
        # 422 — server-side feature extraction rejected the validated input
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        await session.rollback()
        logger.error(f"Application processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Application processing failed: {str(e)}"
        )


def _synthesize_application_payload(simple: SimpleLoanApplicationRequest) -> dict:
    """Build a LoanApplicationRequest-shaped dict from the simpler frontend form.

    Synthesises a plausible 90-day SMS/contact/location/behavioral payload from
    monthly_income and monthly_expenses so the existing feature pipeline can
    run server-side. The v1 model was trained on this shape; in Phase 3 the
    Home Credit retrain replaces this with a real feature schema.
    """
    from datetime import timedelta
    import uuid

    months = 3
    txns: list[dict] = []
    now = datetime.utcnow()

    # `source` is what the feature extractors actually read (legacy column from
    # the synthetic training set). `merchant_category` stays in the public schema
    # for API stability. They carry the same semantic data here.
    income_sources = ["Salary", "Freelance"]  # >1 distinct gives a non-zero income_source_diversity
    debit_categories = [
        # essential (recognized by expense + discipline extractors)
        "Electricity Bill", "Mobile Recharge", "Grocery", "Loan EMI",
        # mixed essentials/discretionary
        "Insurance", "Medical",
        # discretionary
        "Restaurant", "Shopping",
    ]

    # Credits: one per month from the primary source, plus one occasional secondary
    for m in range(months):
        src = income_sources[0] if m % 2 == 0 else income_sources[1]
        txns.append({
            "transaction_id": f"TXN_C_{m}_{uuid.uuid4().hex[:6]}",
            "timestamp": (now - timedelta(days=30 * m)).isoformat(),
            "amount": float(simple.monthly_income),
            "transaction_type": "credit",
            "merchant_category": src,
            "source": src,
            "is_credit": True,
        })

    # Debits: spread across recognised categories
    n_debits_per_month = len(debit_categories)
    for m in range(months):
        for d, category in enumerate(debit_categories):
            day_offset = 30 * m + (d * 3 + 2)
            amt = simple.monthly_expenses / n_debits_per_month
            txns.append({
                "transaction_id": f"TXN_D_{m}_{d}_{uuid.uuid4().hex[:6]}",
                "timestamp": (now - timedelta(days=day_offset)).isoformat(),
                "amount": float(amt),
                "transaction_type": "debit",
                "merchant_category": category,
                "source": category,
                "is_credit": False,
            })

    is_urban = simple.pincode[:1] in ("1", "4", "5", "6", "7")  # crude metro/non-metro heuristic
    return {
        "user_id": f"USER_{simple.mobile}",
        "loan_amount": simple.loan_amount,
        "loan_purpose": simple.loan_purpose,
        "sms_transactions": txns,
        "contact_metadata": {
            "total_contacts": 120,
            "family_contacts": 15,
            "business_contacts": 40,
            "government_contacts": 3,
            "avg_call_duration": 150.0,
            "contact_diversity_score": 0.65,
        },
        "location_pattern": {
            "unique_locations": 4,
            "home_location": {"lat": 28.6139, "lon": 77.2090},
            "travel_radius_km": 18.0,
            "area_type": "urban" if is_urban else "rural",
            "location_stability_score": 0.78,
        },
        "behavioral_data": {
            "app_usage_hours_per_day": 3.5,
            "night_activity_ratio": 0.12,
            "gambling_indicators": 0,
            "financial_app_usage": True,
            "literacy_score": 0.7,
        },
    }


# --- lifecycle constants ---
# Real loans take days to decide. We compress to ~15s so the UX feels like a
# real product (not "instant decision") without being painful to wait through.
LIFECYCLE_SUBMITTED_TO_REVIEW_SECONDS = 5
LIFECYCLE_REVIEW_TO_DECIDED_SECONDS = 10
INITIAL_STATUS = "submitted"
REVIEW_STATUS = "under_review"
DECIDED_STATUS = "decided"
ERROR_STATUS = "rejected"


async def _run_application_lifecycle(application_db_id: str, public_application_id: str) -> None:
    """Background-task lifecycle worker.

    Runs in its own DB session (the request session is already closed by the
    time this fires). Walks the application through submitted -> under_review
    -> decided, recording each transition as an event. Inference happens at
    the decided transition. Any failure transitions to rejected with a reason.
    """
    # Lazy import to avoid circulars at startup (db_manager is imported in main module scope anyway,
    # but the background task needs its OWN session, not the per-request one)
    from ..database.connection import db_manager
    import pandas as pd

    async def _new_session():
        if not db_manager._connected:
            await db_manager.connect()
        return db_manager.session_factory()

    # --- transition 1: submitted -> under_review ---
    await asyncio.sleep(LIFECYCLE_SUBMITTED_TO_REVIEW_SECONDS)
    session = await _new_session()
    try:
        await ApplicationRepository.update_status(
            session, public_application_id, REVIEW_STATUS,
        )
        await ApplicationStatusRepository.record_transition(
            session,
            application_id=application_db_id,
            from_status=INITIAL_STATUS,
            to_status=REVIEW_STATUS,
            actor_type="system",
            reason="automated_review_started",
        )
        await session.commit()
    finally:
        await session.close()

    # --- transition 2: under_review -> decided (with inference) ---
    await asyncio.sleep(LIFECYCLE_REVIEW_TO_DECIDED_SECONDS)
    session = await _new_session()
    try:
        app = await ApplicationRepository.get_by_id(session, public_application_id)
        if not app:
            logger.error(f"Background task: application {public_application_id} vanished mid-lifecycle")
            return

        # Re-hydrate the payload that was persisted at submit time
        app_data = {
            'application_id': public_application_id,
            'user_id': app.user_id,
            'sms_transactions': app.sms_transactions or [],
            'contact_metadata': app.contact_metadata or {},
            'location_pattern': app.location_pattern or {},
            'behavioral_data': app.behavioral_data or {},
        }

        try:
            features = feature_pipeline.extract_features(app_data)
            metadata_cols = ['application_id', 'user_id']
            X = pd.DataFrame([{k: v for k, v in features.items() if k not in metadata_cols}])
            result = predictor.predict(X, return_confidence=True)
        except FeatureExtractionError as e:
            # Bad data — transition to rejected with the extractor name so admin can debug
            await ApplicationRepository.update_status(session, public_application_id, ERROR_STATUS)
            await ApplicationStatusRepository.record_transition(
                session,
                application_id=application_db_id,
                from_status=REVIEW_STATUS,
                to_status=ERROR_STATUS,
                actor_type="system",
                reason=f"feature_extraction_failed: {e.extractor}",
            )
            await session.commit()
            logger.warning(f"Lifecycle rejected {public_application_id}: {e}")
            return
        except Exception as e:
            # Capture both type AND a short str(e) so the failure mode survives the
            # lossy structured-logger / cp1252-stdout path on Windows. Keep ASCII-only
            # in the reason so DB JSON storage never trips on encoding.
            err_msg = str(e).encode('ascii', 'replace').decode('ascii')[:250]
            await ApplicationRepository.update_status(session, public_application_id, ERROR_STATUS)
            await ApplicationStatusRepository.record_transition(
                session,
                application_id=application_db_id,
                from_status=REVIEW_STATUS,
                to_status=ERROR_STATUS,
                actor_type="system",
                reason=f"inference_error: {e.__class__.__name__}: {err_msg}",
            )
            await session.commit()
            # Avoid exc_info=True here: the trace can contain non-ASCII frames that
            # the structured logger sometimes fails to encode on Windows consoles.
            logger.error(
                "Lifecycle inference failed",
                application_id=public_application_id,
                error_type=e.__class__.__name__,
                error=err_msg,
            )
            return

        # Persist features (numeric only — strip the metadata columns the pipeline
        # tacks on, since extracted_features in the response schema is Dict[str, float])
        metadata_keys = {'application_id', 'user_id'}
        numeric_features = {k: float(v) for k, v in features.items() if k not in metadata_keys}
        app.extracted_features = numeric_features

        # Feed the drift detector: rolling buffer of recent inputs
        recent_features_cache.record(numeric_features)
        app.processing_time_ms = result.get('prediction_time_ms', 0.0)
        app.processed_at = datetime.utcnow()
        app.status = DECIDED_STATUS

        prediction_id = f"PRED_{uuid.uuid4().hex[:12].upper()}"
        await PredictionRepository.create(session, {
            'prediction_id': prediction_id,
            'application_id': app.id,
            'profitability_score': result['score'],
            'confidence': result.get('confidence', 0.0),
            'decision': result['decision'],
            'decision_threshold': 0.6,
            'model_version': MODEL_VERSION,
            'model_name': 'profitability_model',
            'prediction_latency_ms': result.get('prediction_time_ms', 0.0),
        })
        await ApplicationStatusRepository.record_transition(
            session,
            application_id=application_db_id,
            from_status=REVIEW_STATUS,
            to_status=DECIDED_STATUS,
            actor_type="system",
            reason=f"decision={result['decision']} score={result['score']:.3f}",
        )
        await session.commit()
        logger.info(
            "Lifecycle decided",
            application_id=public_application_id,
            decision=result['decision'],
            score=round(result['score'], 4),
        )
    finally:
        await session.close()


@app.post(
    "/api/v1/applications/simple",
    response_model=ApplicationSubmittedResponse,
    status_code=status.HTTP_201_CREATED,
)
async def submit_simple_application(
    simple: SimpleLoanApplicationRequest,
    background: BackgroundTasks,
    session: AsyncSession = Depends(get_db),
):
    """Submit a loan application via the simplified frontend form.

    Async-lifecycle flow: persist the application immediately in 'submitted'
    status, schedule a background task that walks it through under_review ->
    decided (with inference) over ~15 seconds, and return right away with the
    application_id so the frontend can poll the timeline.
    """
    payload = _synthesize_application_payload(simple)
    full_request = LoanApplicationRequest(**payload)
    application_id = f"APP_{uuid.uuid4().hex[:12].upper()}"

    # Resolve the user FK. applications.user_id references users.id (UUID PK),
    # NOT the public user_id string. Look up by mobile (which the OTP flow uses
    # to create the user); mint a lightweight shell if no row exists so anonymous
    # curl tests and pre-auth flows still produce a valid FK.
    user = await UserRepository.get_by_phone(session, simple.mobile)
    if user is None:
        user = await UserRepository.create(session, {
            'user_id': f"USER_{simple.mobile}",
            'phone_number': simple.mobile,
            'full_name': simple.name,
            'is_active': True,
        })

    logger.info(
        "Application submitted (async lifecycle)",
        application_id=application_id,
        user_id=user.user_id,
        user_db_id=user.id,
        loan_amount=full_request.loan_amount,
    )

    # model_dump(mode='json') serialises nested datetime objects to ISO strings;
    # plain .dict() leaves them as datetime objects which SQLAlchemy's JSON serializer rejects.
    db_application = await ApplicationRepository.create(session, {
        'application_id': application_id,
        'user_id': user.id,
        'loan_amount': full_request.loan_amount,
        'loan_purpose': full_request.loan_purpose,
        'status': INITIAL_STATUS,
        'sms_transactions': [txn.model_dump(mode='json') for txn in full_request.sms_transactions],
        'contact_metadata': full_request.contact_metadata.model_dump(mode='json'),
        'location_pattern': full_request.location_pattern.model_dump(mode='json'),
        'behavioral_data': full_request.behavioral_data.model_dump(mode='json'),
        'submitted_at': datetime.utcnow(),
    })
    await ApplicationStatusRepository.record_transition(
        session,
        application_id=db_application.id,
        from_status=None,
        to_status=INITIAL_STATUS,
        actor_type="user",
        actor_id=user.user_id,
    )
    await session.commit()

    # Schedule the lifecycle worker. BackgroundTasks runs *after* the response
    # is sent — we pass the DB primary key + public ID so the worker can
    # reopen its own session and look up the row.
    background.add_task(_run_application_lifecycle, db_application.id, application_id)

    return ApplicationSubmittedResponse(
        application_id=application_id,
        status=INITIAL_STATUS,
        submitted_at=db_application.submitted_at,
    )


@app.get(
    "/api/v1/applications/{application_id}/timeline",
    response_model=ApplicationTimelineResponse,
)
async def get_application_timeline(
    application_id: str,
    session: AsyncSession = Depends(get_db),
):
    """Ordered status-transition log for an application.

    Frontend polls this every few seconds while the application is in a
    non-terminal state (submitted / under_review) to render a live timeline.
    """
    application = await ApplicationRepository.get_by_id(session, application_id)
    if not application:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Application {application_id} not found",
        )

    events = await ApplicationStatusRepository.get_timeline(session, application.id)
    return ApplicationTimelineResponse(
        application_id=application_id,
        current_status=application.status,
        events=[
            ApplicationTimelineEvent(
                from_status=e.from_status,
                to_status=e.to_status,
                actor_type=e.actor_type,
                actor_id=e.actor_id,
                reason=e.reason,
                occurred_at=e.occurred_at,
            )
            for e in events
        ],
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


from fastapi import Query as _Query


@app.get("/api/v1/predictions/{application_id}/explain", response_model=ExplanationResponse)
async def explain_prediction(
    application_id: str,
    language: str = _Query("en", regex="^(en|hi)$", description="en | hi"),
    session: AsyncSession = Depends(get_db),
):
    """SHAP-based explanation + Gemini natural-language narration.

    Fetches the stored extracted_features for the application, runs the SHAP
    TreeExplainer, then hands the top contributors to the Gemini Flash layer
    to produce a plain-English (or Hindi) narration. Falls back to a
    deterministic template if Gemini is unavailable — endpoint never 500s on
    LLM failure.
    """
    if not predictor or not explainer:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Explainer not available. Model or explainer not loaded."
        )

    application = await ApplicationRepository.get_by_id(session, application_id)
    if not application:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Application {application_id} not found"
        )

    stored_features = application.extracted_features or {}
    if not stored_features:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Application {application_id} has no stored features to explain"
        )

    try:
        import pandas as pd

        metadata_cols = {'application_id', 'user_id'}
        feature_dict = {k: v for k, v in stored_features.items() if k not in metadata_cols}
        features_df = pd.DataFrame([feature_dict])
        features_df = predictor._validate_features(features_df)

        explanation = explainer.explain_prediction(features_df, top_n=10)

        top_contributors = [
            FeatureContribution(
                feature=contrib['feature'],
                value=contrib['value'],
                contribution=contrib['shap_value'],
                importance=contrib['abs_shap_value'],
            )
            for contrib in explanation['top_contributions']
        ]

        # Hand the top contributors to Gemini for plain-language narration.
        # Shape we send LLM matches the explainer's dict shape so the LLM
        # module isn't coupled to FastAPI / SHAP types.
        llm_input = [
            {"feature": c['feature'], "contribution": c['shap_value']}
            for c in explanation['top_contributions']
        ]
        nle = llm_explain_decision(
            score=float(explanation['prediction']),
            decision=str(explanation['decision']),
            top_features=llm_input,
            language=language,  # type: ignore[arg-type]
        )

        return ExplanationResponse(
            application_id=application_id,
            profitability_score=explanation['prediction'],
            decision=DecisionEnum(explanation['decision']),
            base_value=explanation['base_value'],
            top_contributors=top_contributors,
            explanation_timestamp=datetime.utcnow(),
            natural_language=NaturalLanguageExplanationModel(
                text=nle.text,
                suggestion=nle.suggestion,
                language=nle.language,
                source=nle.source,
                cached=nle.cached,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explanation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation failed: {str(e)}"
        )


@app.get("/api/v1/predictions/{application_id}/counterfactual", response_model=CounterfactualResponse)
async def counterfactual_explanation(
    application_id: str,
    language: str = _Query("en", regex="^(en|hi)$"),
    session: AsyncSession = Depends(get_db),
):
    """'What would need to change for this decision to flip toward approve?'

    Runs a greedy 1-D search over a curated whitelist of *actionable* features
    (income, expenses, savings ratio, bill timeliness, discipline scores), then
    narrates the result via Gemini. Returns reachable=False with an empty
    changes list if no realistic set of changes flips the decision.
    """
    if not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded.",
        )

    application = await ApplicationRepository.get_by_id(session, application_id)
    if not application:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Application {application_id} not found",
        )
    stored = application.extracted_features or {}
    if not stored:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Application {application_id} has no stored features",
        )

    try:
        metadata_keys = {'application_id', 'user_id'}
        features = {k: float(v) for k, v in stored.items() if k not in metadata_keys}
        cf = find_counterfactual(predictor, features)

        # Find the latest prediction to know what decision band we started in
        preds = await PredictionRepository.get_by_application(session, application.id)
        starting_decision = preds[0].decision if preds else "manual_review"

        nle = llm_narrate_counterfactual(
            score=cf['starting_score'],
            decision=starting_decision,
            changes=cf['changes'],
            language=language,  # type: ignore[arg-type]
        )

        return CounterfactualResponse(
            application_id=application_id,
            starting_score=cf['starting_score'],
            final_score=cf['final_score'],
            reachable=cf['reachable'],
            changes=[
                CounterfactualChange(
                    feature=c['feature'],
                    display_label=c['display_label'],
                    display_unit=c.get('display_unit', ''),
                    current=c['current'],
                    suggested=c['suggested'],
                    delta_score=c['delta_score'],
                    new_score=c['new_score'],
                )
                for c in cf['changes']
            ],
            natural_language=NaturalLanguageExplanationModel(
                text=nle.text, suggestion=nle.suggestion, language=nle.language,
                source=nle.source, cached=nle.cached,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Counterfactual failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Counterfactual failed: {str(e)}",
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

                app_data = {
                    'application_id': application_id,
                    'user_id': app_request.user_id,
                    'sms_transactions': [txn.model_dump(mode='json') for txn in app_request.sms_transactions],
                    'contact_metadata': app_request.contact_metadata.model_dump(mode='json'),
                    'location_pattern': app_request.location_pattern.model_dump(mode='json'),
                    'behavioral_data': app_request.behavioral_data.model_dump(mode='json'),
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
