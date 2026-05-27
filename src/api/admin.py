"""Admin (lender/operator) API.

Every route requires the caller's JWT to belong to a user with `role='admin'`
(see `require_admin` dependency). Promote a user via `python scripts/promote_admin.py`.

Endpoints:
- GET  /admin/metrics/overview                — predictions today, score histogram, p95 latency
- GET  /admin/applications                    — paginated queue, status-filterable
- GET  /admin/applications/{application_id}   — full drill-down (applicant, features, SHAP, timeline)
- POST /admin/applications/{application_id}/override — admin overrides decision with audit
- GET  /admin/drift                           — per-feature PSI vs training baseline
"""

from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth.dependencies import require_admin
from ..cache import recent_features as recent_features_cache
from ..database.connection import get_db
from ..database.models import Application, Prediction, User
from ..database.repositories import (
    ApplicationRepository,
    ApplicationStatusRepository,
    PredictionRepository,
)
from ..models.drift_detector import compute_feature_drift
from ..utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


# --- Schemas (inline — admin-specific, no need to bloat shared schemas.py) ---


class ScoreBucket(BaseModel):
    range_low: float
    range_high: float
    count: int


class DecisionCount(BaseModel):
    decision: str
    count: int


class AdminMetricsOverview(BaseModel):
    """Top-line operator KPIs over the last 24 hours.

    Model-performance fields (total_predictions, avg_score, avg_confidence,
    avg_latency_ms, p95_latency_ms, decisions, score_histogram) reflect only
    real model predictions — admin overrides are tracked separately below so
    a high override rate doesn't masquerade as model behaviour.
    """
    window_hours: int = 24
    total_predictions: int
    avg_score: float
    avg_confidence: float
    avg_latency_ms: float
    p95_latency_ms: float
    decisions: List[DecisionCount]
    score_histogram: List[ScoreBucket]
    # Operator activity — separated from model metrics
    override_count: int                        # admin-override predictions in window
    override_decisions: List[DecisionCount]    # breakdown of what operators decided
    # NOT window-bound: a NOW snapshot of pipeline state
    pending_review_count: int                  # applications stuck in submitted/under_review
    errored_count: int                         # applications in 'rejected' state with no prediction (inference errors)
    recent_buffer_size: int                    # drift detector buffer
    drift_baseline_available: bool


class AdminApplicationSummary(BaseModel):
    application_id: str
    user_id: str
    user_phone: Optional[str] = None
    loan_amount: float
    loan_purpose: str
    status: str
    submitted_at: datetime
    decision: Optional[str] = None
    score: Optional[float] = None


class AdminApplicationsListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    applications: List[AdminApplicationSummary]


class AdminApplicationDetail(BaseModel):
    application_id: str
    user_id: str
    user_phone: Optional[str] = None
    user_full_name: Optional[str] = None
    loan_amount: float
    loan_purpose: str
    status: str
    submitted_at: datetime
    processed_at: Optional[datetime] = None
    processing_time_ms: Optional[float] = None
    extracted_features: Optional[dict] = None
    latest_prediction: Optional[dict] = None
    timeline: List[dict]


class AdminOverrideRequest(BaseModel):
    decision: str = Field(..., description="approve | reject | manual_review")
    reason: str = Field(..., min_length=2, max_length=500, description="Audit-trail reason for the override")


class AdminOverrideResponse(BaseModel):
    application_id: str
    new_status: str
    overridden_decision: str
    overridden_by: str
    occurred_at: datetime


class DriftFeature(BaseModel):
    feature: str
    psi: Optional[float] = None
    severity: str
    n_current: int
    baseline_available: bool


class AdminDriftResponse(BaseModel):
    n_recent_inputs: int
    baseline_available: bool
    summary: dict                             # {stable, moderate, significant, no_data counts}
    features: List[DriftFeature]


# --- Helpers ---


_TERMINAL_DECISIONS = {"approve", "reject", "manual_review"}
_OVERRIDABLE_FROM_STATUSES = {"under_review", "decided", "submitted"}


async def _user_phone_map(session: AsyncSession, user_db_ids: List[str]) -> dict:
    """Bulk-lookup phone numbers + names for a set of user DB ids."""
    if not user_db_ids:
        return {}
    rows = await session.execute(
        select(User.id, User.phone_number, User.full_name).where(User.id.in_(user_db_ids))
    )
    return {r[0]: {"phone": r[1], "name": r[2]} for r in rows.all()}


# --- Endpoints ---


@router.get("/metrics/overview", response_model=AdminMetricsOverview)
async def admin_metrics_overview(
    _admin: User = Depends(require_admin),
    session: AsyncSession = Depends(get_db),
    window_hours: int = Query(24, ge=1, le=168),
):
    """Lender KPIs over the last `window_hours` (default 24h)."""
    from ..api.main import predictor  # avoid circular import at module load

    since = datetime.utcnow() - timedelta(hours=window_hours)

    # Model-perf metrics exclude admin overrides — overrides write synthetic
    # rows with score=1.0/0.5/0.0, conf=1.0, latency=0.0 that would otherwise
    # distort avg_score, avg_confidence, the P95, the decision counts, and the
    # histogram. Operator activity is reported separately further down.
    OVERRIDE_TAG = "admin-override"
    model_only = (
        (Prediction.prediction_timestamp >= since)
        & (Prediction.model_version != OVERRIDE_TAG)
    )

    # Total + averages from the predictions table (model predictions only)
    totals = await session.execute(
        select(
            func.count(Prediction.id),
            func.avg(Prediction.profitability_score),
            func.avg(Prediction.confidence),
            func.avg(Prediction.prediction_latency_ms),
        ).where(model_only)
    )
    total, avg_score, avg_conf, avg_lat = totals.one()

    # P95 latency: percentile_cont is portable across PG versions
    p95_row = await session.execute(
        select(func.percentile_cont(0.95).within_group(Prediction.prediction_latency_ms))
        .where(model_only)
    )
    p95_lat = p95_row.scalar()

    # Model decision breakdown (overrides excluded)
    decisions_q = await session.execute(
        select(Prediction.decision, func.count(Prediction.id))
        .where(model_only)
        .group_by(Prediction.decision)
    )
    decisions = [DecisionCount(decision=row[0], count=row[1]) for row in decisions_q.all()]

    # Score histogram (10 bins, 0.0-1.0). Done in Python over the recent set —
    # at <10K predictions this is cheap and avoids DB-vendor-specific bucketing.
    scores_q = await session.execute(
        select(Prediction.profitability_score).where(model_only)
    )
    scores = [float(r[0]) for r in scores_q.all()]
    histogram: List[ScoreBucket] = []
    for i in range(10):
        lo = i / 10
        hi = (i + 1) / 10
        count = sum(1 for s in scores if (lo <= s < hi) or (i == 9 and s == 1.0))
        histogram.append(ScoreBucket(range_low=round(lo, 1), range_high=round(hi, 1), count=count))

    # Operator overrides — same window, counted on their own
    override_only = (
        (Prediction.prediction_timestamp >= since)
        & (Prediction.model_version == OVERRIDE_TAG)
    )
    override_total_row = await session.execute(
        select(func.count(Prediction.id)).where(override_only)
    )
    override_count = override_total_row.scalar() or 0
    override_decisions_q = await session.execute(
        select(Prediction.decision, func.count(Prediction.id))
        .where(override_only)
        .group_by(Prediction.decision)
    )
    override_decisions = [
        DecisionCount(decision=row[0], count=row[1]) for row in override_decisions_q.all()
    ]

    # NOT window-bound — these are NOW snapshots of pipeline state.
    pending_q = await session.execute(
        select(func.count(Application.id)).where(
            Application.status.in_(["submitted", "under_review"])
        )
    )
    pending = pending_q.scalar() or 0
    errored_q = await session.execute(
        select(func.count(Application.id)).where(Application.status == "rejected")
    )
    errored = errored_q.scalar() or 0

    return AdminMetricsOverview(
        window_hours=window_hours,
        total_predictions=total or 0,
        avg_score=round(float(avg_score or 0.0), 4),
        avg_confidence=round(float(avg_conf or 0.0), 4),
        avg_latency_ms=round(float(avg_lat or 0.0), 2),
        p95_latency_ms=round(float(p95_lat or 0.0), 2),
        decisions=decisions,
        score_histogram=histogram,
        override_count=override_count,
        override_decisions=override_decisions,
        pending_review_count=pending,
        errored_count=errored,
        recent_buffer_size=recent_features_cache.size(),
        drift_baseline_available=bool(predictor and predictor.feature_quantiles),
    )


@router.get("/applications", response_model=AdminApplicationsListResponse)
async def admin_list_applications(
    _admin: User = Depends(require_admin),
    session: AsyncSession = Depends(get_db),
    status_filter: Optional[str] = Query(None, description="submitted | under_review | decided | rejected | disbursed"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Paginated queue. Default sort is most-recent submission first."""
    base = select(Application)
    count_base = select(func.count(Application.id))
    if status_filter:
        base = base.where(Application.status == status_filter)
        count_base = count_base.where(Application.status == status_filter)

    total_row = await session.execute(count_base)
    total = total_row.scalar() or 0

    rows = await session.execute(
        base.order_by(Application.submitted_at.desc()).limit(limit).offset(offset)
    )
    applications = rows.scalars().all()

    phone_map = await _user_phone_map(session, [a.user_id for a in applications])

    # Fetch latest prediction per application in one query
    pred_rows = await session.execute(
        select(Prediction)
        .where(Prediction.application_id.in_([a.id for a in applications]))
        .order_by(Prediction.prediction_timestamp.desc())
    )
    latest_pred: dict = {}
    for p in pred_rows.scalars().all():
        latest_pred.setdefault(p.application_id, p)

    summaries = []
    for app in applications:
        p = latest_pred.get(app.id)
        u = phone_map.get(app.user_id, {})
        summaries.append(AdminApplicationSummary(
            application_id=app.application_id,
            user_id=app.user_id,
            user_phone=u.get("phone"),
            loan_amount=app.loan_amount,
            loan_purpose=app.loan_purpose,
            status=app.status,
            submitted_at=app.submitted_at,
            decision=p.decision if p else None,
            score=float(p.profitability_score) if p else None,
        ))

    return AdminApplicationsListResponse(
        total=total,
        limit=limit,
        offset=offset,
        applications=summaries,
    )


@router.get("/applications/{application_id}", response_model=AdminApplicationDetail)
async def admin_application_detail(
    application_id: str,
    _admin: User = Depends(require_admin),
    session: AsyncSession = Depends(get_db),
):
    """Full audit view for one application: applicant, features, prediction, timeline."""
    app = await ApplicationRepository.get_by_id(session, application_id)
    if not app:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Application {application_id} not found")

    user_row = await session.execute(select(User).where(User.id == app.user_id))
    user = user_row.scalar_one_or_none()

    preds = await PredictionRepository.get_by_application(session, app.id)
    latest = preds[0] if preds else None
    latest_dict = None
    if latest:
        latest_dict = {
            "profitability_score": float(latest.profitability_score),
            "confidence": float(latest.confidence),
            "decision": latest.decision,
            "model_version": latest.model_version,
            "prediction_latency_ms": latest.prediction_latency_ms,
            "prediction_timestamp": latest.prediction_timestamp.isoformat(),
        }

    timeline = await ApplicationStatusRepository.get_timeline(session, app.id)
    timeline_dicts = [
        {
            "from_status": e.from_status,
            "to_status": e.to_status,
            "actor_type": e.actor_type,
            "actor_id": e.actor_id,
            "reason": e.reason,
            "occurred_at": e.occurred_at.isoformat(),
        }
        for e in timeline
    ]

    return AdminApplicationDetail(
        application_id=app.application_id,
        user_id=app.user_id,
        user_phone=user.phone_number if user else None,
        user_full_name=user.full_name if user else None,
        loan_amount=app.loan_amount,
        loan_purpose=app.loan_purpose,
        status=app.status,
        submitted_at=app.submitted_at,
        processed_at=app.processed_at,
        processing_time_ms=app.processing_time_ms,
        extracted_features=app.extracted_features,
        latest_prediction=latest_dict,
        timeline=timeline_dicts,
    )


@router.post("/applications/{application_id}/override", response_model=AdminOverrideResponse)
async def admin_override_decision(
    application_id: str,
    body: AdminOverrideRequest,
    admin: User = Depends(require_admin),
    session: AsyncSession = Depends(get_db),
):
    """Admin overrides the model's decision. Records an immutable audit event."""
    if body.decision not in _TERMINAL_DECISIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"decision must be one of {sorted(_TERMINAL_DECISIONS)}",
        )

    app = await ApplicationRepository.get_by_id(session, application_id)
    if not app:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Application {application_id} not found")

    if app.status not in _OVERRIDABLE_FROM_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot override an application in status '{app.status}'",
        )

    # Map decision to a status. approve/reject -> their own terminal state;
    # manual_review keeps the application in under_review with a noted reason.
    new_status = "decided" if body.decision in ("approve", "reject") else "under_review"
    from_status = app.status

    app.status = new_status
    app.processed_at = datetime.utcnow()

    # Insert a fresh Prediction row reflecting the override so the user-facing
    # detail page shows the new decision. We tag model_version='admin-override'
    # so downstream analytics can separate model decisions from human ones.
    import uuid as _uuid
    await PredictionRepository.create(session, {
        "prediction_id": f"OVR_{_uuid.uuid4().hex[:12].upper()}",
        "application_id": app.id,
        "profitability_score": 1.0 if body.decision == "approve" else 0.0 if body.decision == "reject" else 0.5,
        "confidence": 1.0,
        "decision": body.decision,
        "decision_threshold": 0.6,
        "model_version": "admin-override",
        "model_name": "admin-override",
        "prediction_latency_ms": 0.0,
    })

    event = await ApplicationStatusRepository.record_transition(
        session,
        application_id=app.id,
        from_status=from_status,
        to_status=new_status,
        actor_type="admin",
        actor_id=admin.user_id,
        reason=f"override decision={body.decision}: {body.reason}",
    )
    await session.commit()

    logger.info(
        "Admin override",
        application_id=application_id,
        admin=admin.user_id,
        decision=body.decision,
    )

    return AdminOverrideResponse(
        application_id=application_id,
        new_status=new_status,
        overridden_decision=body.decision,
        overridden_by=admin.user_id,
        occurred_at=event.occurred_at,
    )


@router.get("/drift", response_model=AdminDriftResponse)
async def admin_drift(
    _admin: User = Depends(require_admin),
):
    """Per-feature PSI between training distribution and recent live inputs."""
    from ..api.main import predictor  # avoid circular import

    if not predictor or not predictor.feature_names:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded — drift unavailable.",
        )

    recent = recent_features_cache.snapshot()
    features = compute_feature_drift(
        reference_quantiles=predictor.feature_quantiles,
        current_rows=recent,
        feature_names=predictor.feature_names,
    )

    summary = {"stable": 0, "moderate": 0, "significant": 0, "unknown": 0, "no_data": 0}
    for f in features:
        summary[f["severity"]] = summary.get(f["severity"], 0) + 1

    return AdminDriftResponse(
        n_recent_inputs=len(recent),
        baseline_available=bool(predictor.feature_quantiles),
        summary=summary,
        features=[DriftFeature(**f) for f in features],
    )
