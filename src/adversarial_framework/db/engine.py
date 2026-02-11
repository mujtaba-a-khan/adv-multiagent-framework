"""Async SQLAlchemy engine and session factory for PostgreSQL.

Usage::

    from adversarial_framework.db.engine import get_session

    async with get_session() as session:
        result = await session.execute(select(Experiment))
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from adversarial_framework.config.settings import get_settings

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Return the global async engine, creating it on first call."""
    global _engine
    if _engine is None:
        settings = get_settings()
        if not settings.database_url:
            raise RuntimeError(
                "ADV_DATABASE_URL is not set. Set it to your PostgreSQL connection string."
            )
        _engine = create_async_engine(
            settings.database_url,
            echo=settings.environment == "development",
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=300,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the global session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield a transactional async session, auto-committing on success."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def dispose_engine() -> None:
    """Dispose the engine connection pool (call on shutdown)."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
