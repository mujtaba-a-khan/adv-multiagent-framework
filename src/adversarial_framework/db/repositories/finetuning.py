"""Repository for FineTuningJob CRUD operations."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from adversarial_framework.db.models import FineTuningJob


class FineTuningJobRepository:
    """Async CRUD for fine-tuning jobs."""

    def __init__(self, session: AsyncSession) -> None:
        self.db = session

    async def create(self, **kwargs: object) -> FineTuningJob:
        job = FineTuningJob(**kwargs)
        self.db.add(job)
        await self.db.flush()
        return job

    async def get(self, job_id: uuid.UUID) -> FineTuningJob | None:
        return await self.db.get(FineTuningJob, job_id)

    async def list_all(
        self,
        offset: int = 0,
        limit: int = 50,
        status: str | None = None,
    ) -> tuple[Sequence[FineTuningJob], int]:
        stmt = (
            select(FineTuningJob)
            .order_by(FineTuningJob.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        count_stmt = select(func.count()).select_from(FineTuningJob)

        if status is not None:
            stmt = stmt.where(FineTuningJob.status == status)
            count_stmt = count_stmt.where(
                FineTuningJob.status == status
            )

        result = await self.db.execute(stmt)
        jobs = result.scalars().all()

        count_result = await self.db.execute(count_stmt)
        total = count_result.scalar_one()

        return jobs, total

    async def update_status(
        self,
        job_id: uuid.UUID,
        status: str,
        *,
        error_message: str | None = None,
    ) -> None:
        now = datetime.now(UTC)
        values: dict[str, object] = {"status": status}

        if status == "running":
            values["started_at"] = now
        elif status in ("completed", "failed", "cancelled"):
            values["completed_at"] = now

        if error_message is not None:
            values["error_message"] = error_message

        await self.db.execute(
            update(FineTuningJob)
            .where(FineTuningJob.id == job_id)
            .values(**values)
        )
        self.db.expire_all()

    async def update_progress(
        self,
        job_id: uuid.UUID,
        *,
        progress_pct: float,
        current_step: str,
        peak_memory_gb: float | None = None,
        total_duration_seconds: float | None = None,
    ) -> None:
        values: dict[str, object] = {
            "progress_pct": progress_pct,
            "current_step": current_step,
        }
        if peak_memory_gb is not None:
            values["peak_memory_gb"] = peak_memory_gb
        if total_duration_seconds is not None:
            values["total_duration_seconds"] = total_duration_seconds

        await self.db.execute(
            update(FineTuningJob)
            .where(FineTuningJob.id == job_id)
            .values(**values)
        )

    async def append_log(
        self, job_id: uuid.UUID, entry: dict
    ) -> None:
        job = await self.get(job_id)
        if job is not None:
            logs = list(job.logs) if job.logs else []
            logs.append(entry)
            await self.db.execute(
                update(FineTuningJob)
                .where(FineTuningJob.id == job_id)
                .values(logs=logs)
            )

    async def delete(self, job_id: uuid.UUID) -> bool:
        job = await self.get(job_id)
        if job is None:
            return False
        await self.db.delete(job)
        await self.db.flush()
        return True
