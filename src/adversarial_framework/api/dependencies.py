"""FastAPI dependency injection: DB sessions, repositories, providers."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from adversarial_framework.agents.target.providers.ollama import OllamaProvider
from adversarial_framework.config.settings import Settings, get_settings
from adversarial_framework.db.engine import get_session_factory
from adversarial_framework.db.repositories.abliteration_prompts import (
    AbliterationPromptRepository,
)
from adversarial_framework.db.repositories.experiments import ExperimentRepository
from adversarial_framework.db.repositories.finetuning import FineTuningJobRepository
from adversarial_framework.db.repositories.playground import PlaygroundRepository
from adversarial_framework.db.repositories.sessions import SessionRepository
from adversarial_framework.db.repositories.turns import TurnRepository


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


DbSession = Annotated[AsyncSession, Depends(get_db)]


def get_experiment_repo(session: DbSession) -> ExperimentRepository:
    return ExperimentRepository(session)


def get_session_repo(session: DbSession) -> SessionRepository:
    return SessionRepository(session)


def get_turn_repo(session: DbSession) -> TurnRepository:
    return TurnRepository(session)


def get_finetuning_repo(session: DbSession) -> FineTuningJobRepository:
    return FineTuningJobRepository(session)


def get_abliteration_prompt_repo(
    session: DbSession,
) -> AbliterationPromptRepository:
    return AbliterationPromptRepository(session)


def get_playground_repo(session: DbSession) -> PlaygroundRepository:
    return PlaygroundRepository(session)


SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_ollama_provider(settings: SettingsDep) -> OllamaProvider:
    return OllamaProvider(base_url=settings.ollama_base_url)

# Annotated DI aliases for endpoint signatures
ExperimentRepoDep = Annotated[
    ExperimentRepository, Depends(get_experiment_repo)
]
SessionRepoDep = Annotated[SessionRepository, Depends(get_session_repo)]
TurnRepoDep = Annotated[TurnRepository, Depends(get_turn_repo)]
FineTuningRepoDep = Annotated[
    FineTuningJobRepository, Depends(get_finetuning_repo)
]
AbliterationPromptRepoDep = Annotated[
    AbliterationPromptRepository, Depends(get_abliteration_prompt_repo)
]
PlaygroundRepoDep = Annotated[
    PlaygroundRepository, Depends(get_playground_repo)
]
OllamaProviderDep = Annotated[
    OllamaProvider, Depends(get_ollama_provider)
]
