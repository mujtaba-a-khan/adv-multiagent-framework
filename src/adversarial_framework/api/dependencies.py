"""FastAPI dependency injection: DB sessions, repositories, providers."""

from __future__ import annotations

from typing import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from adversarial_framework.db.engine import get_session_factory
from adversarial_framework.db.repositories.experiments import ExperimentRepository
from adversarial_framework.db.repositories.finetuning import FineTuningJobRepository
from adversarial_framework.db.repositories.sessions import SessionRepository
from adversarial_framework.db.repositories.turns import TurnRepository
from adversarial_framework.agents.target.providers.ollama import OllamaProvider
from adversarial_framework.config.settings import Settings, get_settings


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a transactional DB session per request."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_experiment_repo(
    session: AsyncSession = Depends(get_db),
) -> ExperimentRepository:
    return ExperimentRepository(session)


def get_session_repo(
    session: AsyncSession = Depends(get_db),
) -> SessionRepository:
    return SessionRepository(session)


def get_turn_repo(
    session: AsyncSession = Depends(get_db),
) -> TurnRepository:
    return TurnRepository(session)


def get_finetuning_repo(
    session: AsyncSession = Depends(get_db),
) -> FineTuningJobRepository:
    return FineTuningJobRepository(session)


def get_ollama_provider(
    settings: Settings = Depends(get_settings),
) -> OllamaProvider:
    return OllamaProvider(base_url=settings.ollama_base_url)
