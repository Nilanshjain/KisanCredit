"""Database connection and session management with async support."""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool, AsyncAdaptedQueuePool
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import time

from ..utils.logger import get_logger
from ..utils.config import settings

logger = get_logger(__name__)


class DatabaseManager:
    """Async database connection manager with connection pooling."""

    def __init__(self, database_url: str = None, echo: bool = False):
        """Initialize database manager.

        Args:
            database_url: Database connection URL (uses config if None)
            echo: Whether to echo SQL statements
        """
        self.database_url = database_url or settings.database_url
        self.echo = echo
        self.engine = None
        self.session_factory = None
        self._connected = False

    async def connect(self):
        """Create async engine and session factory."""
        if self._connected:
            logger.warning("Database already connected")
            return

        try:
            logger.info("Connecting to database...")

            # Create async engine with connection pooling
            self.engine = create_async_engine(
                self.database_url,
                echo=self.echo,
                pool_size=20,  # Number of connections to maintain
                max_overflow=10,  # Additional connections when pool is full
                pool_timeout=30,  # Timeout for getting connection from pool
                pool_recycle=3600,  # Recycle connections after 1 hour
                pool_pre_ping=True,  # Verify connections before using
                poolclass=AsyncAdaptedQueuePool,
            )

            # Create session factory
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autocommit=False,
                autoflush=False
            )

            self._connected = True

            logger.info(
                "Database connected",
                pool_size=20,
                max_overflow=10,
                database_url=self._mask_url(self.database_url)
            )

        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    async def disconnect(self):
        """Close database connections."""
        if not self._connected:
            return

        try:
            logger.info("Disconnecting from database...")

            if self.engine:
                await self.engine.dispose()

            self._connected = False
            logger.info("Database disconnected")

        except Exception as e:
            logger.error(f"Database disconnection failed: {e}")
            raise

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session with automatic cleanup.

        Usage:
            async with db_manager.get_session() as session:
                result = await session.execute(query)
        """
        if not self._connected:
            await self.connect()

        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()

    async def health_check(self) -> dict:
        """Check database connection health.

        Returns:
            Dictionary with health status
        """
        if not self._connected:
            return {
                "is_healthy": False,
                "error": "Not connected"
            }

        try:
            start_time = time.time()

            # Test connection with simple query
            async with self.get_session() as session:
                from sqlalchemy import text
                result = await session.execute(text("SELECT 1"))
                await result.fetchone()

            latency = (time.time() - start_time) * 1000

            return {
                "is_healthy": True,
                "latency_ms": round(latency, 2),
                "connected": True,
                "pool_size": self.engine.pool.size() if self.engine else 0,
                "checked_in": self.engine.pool.checkedin() if self.engine else 0
            }

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "is_healthy": False,
                "error": str(e)
            }

    def _mask_url(self, url: str) -> str:
        """Mask password in database URL for logging.

        Args:
            url: Database URL

        Returns:
            Masked URL
        """
        if '@' in url:
            parts = url.split('@')
            if ':' in parts[0]:
                protocol_user = parts[0].rsplit(':', 1)[0]
                return f"{protocol_user}:****@{parts[1]}"
        return url

    async def create_all_tables(self):
        """Create all database tables.

        Note: Use Alembic migrations in production.
        """
        from .models import Base

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("All database tables created")

    async def drop_all_tables(self):
        """Drop all database tables.

        Warning: This will delete all data!
        """
        from .models import Base

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

        logger.warning("All database tables dropped")


# Global database manager instance
db_manager = DatabaseManager()


# Dependency for FastAPI
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions.

    Usage in FastAPI:
        @app.get("/endpoint")
        async def endpoint(db: AsyncSession = Depends(get_db)):
            result = await db.execute(query)
    """
    async with db_manager.get_session() as session:
        yield session
