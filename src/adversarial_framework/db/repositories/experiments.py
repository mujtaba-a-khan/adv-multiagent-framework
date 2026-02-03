"""Repository for Experiment CRUD operations."""

from __future__ import annotations

import uuid
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from adversarial_framework.db.models import Experiment


class ExperimentRepository:
    """Async CRUD for experiments."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, **kwargs: object) -> Experiment:
        experiment = Experiment(**kwargs)
        self.session.add(experiment)
        await self.session.flush()
        return experiment

    async def get(self, experiment_id: uuid.UUID) -> Experiment | None:
        return await self.session.get(Experiment, experiment_id)

    async def get_with_sessions(self, experiment_id: uuid.UUID) -> Experiment | None:
        stmt = (
            select(Experiment)
            .where(Experiment.id == experiment_id)
            .options(selectinload(Experiment.sessions))
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_all(
        self, offset: int = 0, limit: int = 50
    ) -> Sequence[Experiment]:
        stmt = (
            select(Experiment)
            .order_by(Experiment.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def update(
        self, experiment_id: uuid.UUID, **kwargs: object
    ) -> Experiment | None:
        experiment = await self.get(experiment_id)
        if experiment is None:
            return None
        for key, value in kwargs.items():
            setattr(experiment, key, value)
        await self.session.flush()
        await self.session.refresh(experiment)
        return experiment

    async def delete(self, experiment_id: uuid.UUID) -> bool:
        experiment = await self.get(experiment_id)
        if experiment is None:
            return False
        await self.session.delete(experiment)
        await self.session.flush()
        return True
