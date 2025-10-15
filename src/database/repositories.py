"""Repository layer for database operations."""

from sqlalchemy import select, update, delete, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from .models import User, Application, Prediction, AuditLog, ModelMetrics, CacheMetrics
from ..utils.logger import get_logger

logger = get_logger(__name__)


class UserRepository:
    """Repository for User operations."""

    @staticmethod
    async def create(session: AsyncSession, user_data: Dict[str, Any]) -> User:
        """Create a new user."""
        user = User(**user_data)
        session.add(user)
        await session.flush()
        logger.info(f"User created: {user.user_id}")
        return user

    @staticmethod
    async def get_by_id(session: AsyncSession, user_id: str) -> Optional[User]:
        """Get user by ID."""
        result = await session.execute(select(User).where(User.user_id == user_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_phone(session: AsyncSession, phone_number: str) -> Optional[User]:
        """Get user by phone number."""
        result = await session.execute(select(User).where(User.phone_number == phone_number))
        return result.scalar_one_or_none()

    @staticmethod
    async def update(session: AsyncSession, user_id: str, update_data: Dict[str, Any]) -> Optional[User]:
        """Update user."""
        stmt = update(User).where(User.user_id == user_id).values(**update_data)
        await session.execute(stmt)
        return await UserRepository.get_by_id(session, user_id)


class ApplicationRepository:
    """Repository for Application operations."""

    @staticmethod
    async def create(session: AsyncSession, app_data: Dict[str, Any]) -> Application:
        """Create a new application."""
        application = Application(**app_data)
        session.add(application)
        await session.flush()
        logger.info(f"Application created: {application.application_id}")
        return application

    @staticmethod
    async def get_by_id(session: AsyncSession, application_id: str) -> Optional[Application]:
        """Get application by ID."""
        result = await session.execute(
            select(Application).where(Application.application_id == application_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_user(session: AsyncSession, user_id: str, limit: int = 10) -> List[Application]:
        """Get applications by user ID."""
        result = await session.execute(
            select(Application)
            .where(Application.user_id == user_id)
            .order_by(Application.submitted_at.desc())
            .limit(limit)
        )
        return result.scalars().all()

    @staticmethod
    async def update_status(
        session: AsyncSession,
        application_id: str,
        status: str,
        processed_at: Optional[datetime] = None
    ) -> Optional[Application]:
        """Update application status."""
        update_data = {"status": status, "updated_at": datetime.utcnow()}
        if processed_at:
            update_data["processed_at"] = processed_at

        stmt = update(Application).where(
            Application.application_id == application_id
        ).values(**update_data)
        await session.execute(stmt)
        return await ApplicationRepository.get_by_id(session, application_id)

    @staticmethod
    async def get_recent(session: AsyncSession, limit: int = 100) -> List[Application]:
        """Get recent applications."""
        result = await session.execute(
            select(Application)
            .order_by(Application.submitted_at.desc())
            .limit(limit)
        )
        return result.scalars().all()

    @staticmethod
    async def count_by_status(session: AsyncSession, status: str) -> int:
        """Count applications by status."""
        result = await session.execute(
            select(func.count(Application.id)).where(Application.status == status)
        )
        return result.scalar()


class PredictionRepository:
    """Repository for Prediction operations."""

    @staticmethod
    async def create(session: AsyncSession, pred_data: Dict[str, Any]) -> Prediction:
        """Create a new prediction."""
        prediction = Prediction(**pred_data)
        session.add(prediction)
        await session.flush()
        logger.info(f"Prediction created: {prediction.prediction_id}")
        return prediction

    @staticmethod
    async def get_by_id(session: AsyncSession, prediction_id: str) -> Optional[Prediction]:
        """Get prediction by ID."""
        result = await session.execute(
            select(Prediction).where(Prediction.prediction_id == prediction_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_application(session: AsyncSession, application_id: str) -> List[Prediction]:
        """Get predictions for an application."""
        result = await session.execute(
            select(Prediction)
            .where(Prediction.application_id == application_id)
            .order_by(Prediction.prediction_timestamp.desc())
        )
        return result.scalars().all()

    @staticmethod
    async def get_recent(session: AsyncSession, limit: int = 100) -> List[Prediction]:
        """Get recent predictions."""
        result = await session.execute(
            select(Prediction)
            .order_by(Prediction.prediction_timestamp.desc())
            .limit(limit)
        )
        return result.scalars().all()

    @staticmethod
    async def get_statistics(session: AsyncSession, hours: int = 24) -> Dict[str, Any]:
        """Get prediction statistics for last N hours."""
        since = datetime.utcnow() - timedelta(hours=hours)

        # Total predictions
        total_result = await session.execute(
            select(func.count(Prediction.id))
            .where(Prediction.prediction_timestamp >= since)
        )
        total = total_result.scalar()

        # Decisions breakdown
        decisions_result = await session.execute(
            select(
                Prediction.decision,
                func.count(Prediction.id)
            )
            .where(Prediction.prediction_timestamp >= since)
            .group_by(Prediction.decision)
        )
        decisions = {row[0]: row[1] for row in decisions_result.all()}

        # Average score
        avg_score_result = await session.execute(
            select(func.avg(Prediction.profitability_score))
            .where(Prediction.prediction_timestamp >= since)
        )
        avg_score = avg_score_result.scalar() or 0.0

        # Average latency
        avg_latency_result = await session.execute(
            select(func.avg(Prediction.prediction_latency_ms))
            .where(Prediction.prediction_timestamp >= since)
        )
        avg_latency = avg_latency_result.scalar() or 0.0

        return {
            "total_predictions": total,
            "decisions": decisions,
            "avg_profitability_score": round(float(avg_score), 4),
            "avg_latency_ms": round(float(avg_latency), 2),
            "period_hours": hours
        }


class AuditLogRepository:
    """Repository for AuditLog operations."""

    @staticmethod
    async def create(session: AsyncSession, log_data: Dict[str, Any]) -> AuditLog:
        """Create an audit log entry."""
        audit_log = AuditLog(**log_data)
        session.add(audit_log)
        await session.flush()
        return audit_log

    @staticmethod
    async def get_by_application(
        session: AsyncSession,
        application_id: str,
        limit: int = 50
    ) -> List[AuditLog]:
        """Get audit logs for an application."""
        result = await session.execute(
            select(AuditLog)
            .where(AuditLog.application_id == application_id)
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()

    @staticmethod
    async def get_by_event_type(
        session: AsyncSession,
        event_type: str,
        limit: int = 100
    ) -> List[AuditLog]:
        """Get audit logs by event type."""
        result = await session.execute(
            select(AuditLog)
            .where(AuditLog.event_type == event_type)
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()

    @staticmethod
    async def get_recent(session: AsyncSession, limit: int = 100) -> List[AuditLog]:
        """Get recent audit logs."""
        result = await session.execute(
            select(AuditLog)
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()


class MetricsRepository:
    """Repository for metrics operations."""

    @staticmethod
    async def create_model_metrics(session: AsyncSession, metrics_data: Dict[str, Any]) -> ModelMetrics:
        """Create model metrics entry."""
        metrics = ModelMetrics(**metrics_data)
        session.add(metrics)
        await session.flush()
        return metrics

    @staticmethod
    async def create_cache_metrics(session: AsyncSession, metrics_data: Dict[str, Any]) -> CacheMetrics:
        """Create cache metrics entry."""
        metrics = CacheMetrics(**metrics_data)
        session.add(metrics)
        await session.flush()
        return metrics

    @staticmethod
    async def get_model_metrics(
        session: AsyncSession,
        period_type: str = "hourly",
        limit: int = 24
    ) -> List[ModelMetrics]:
        """Get model metrics."""
        result = await session.execute(
            select(ModelMetrics)
            .where(ModelMetrics.period_type == period_type)
            .order_by(ModelMetrics.period_start.desc())
            .limit(limit)
        )
        return result.scalars().all()

    @staticmethod
    async def get_cache_metrics(
        session: AsyncSession,
        hours: int = 24
    ) -> List[CacheMetrics]:
        """Get cache metrics for last N hours."""
        since = datetime.utcnow() - timedelta(hours=hours)
        result = await session.execute(
            select(CacheMetrics)
            .where(CacheMetrics.timestamp >= since)
            .order_by(CacheMetrics.timestamp.desc())
        )
        return result.scalars().all()
