"""Repository for Session CRUD operations."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Sequence

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from adversarial_framework.db.models import Session


class SessionRepository:
    """Async CRUD for sessions."""

    def __init__(self, session: AsyncSession) -> None:
        self.db = session

    async def create(self, experiment_id: uuid.UUID, **kwargs: object) -> Session:
        sess = Session(experiment_id=experiment_id, **kwargs)
        self.db.add(sess)
        await self.db.flush()
        return sess

    async def get(self, session_id: uuid.UUID) -> Session | None:
        return await self.db.get(Session, session_id)

    async def get_with_turns(self, session_id: uuid.UUID) -> Session | None:
        stmt = (
            select(Session)
            .where(Session.id == session_id)
            .options(selectinload(Session.turns))
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_experiment(
        self, experiment_id: uuid.UUID, offset: int = 0, limit: int = 50
    ) -> tuple[Sequence[Session], int]:
        stmt = (
            select(Session)
            .where(Session.experiment_id == experiment_id)
            .order_by(Session.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        sessions = result.scalars().all()

        count_result = await self.db.execute(
            select(func.count()).select_from(Session).where(
                Session.experiment_id == experiment_id
            )
        )
        total = count_result.scalar_one()

        return sessions, total

    async def update_status(
        self, session_id: uuid.UUID, status: str
    ) -> None:
        extras: dict = {}
        if status == "running":
            extras["started_at"] = datetime.now(timezone.utc)
        elif status in ("completed", "failed", "cancelled"):
            extras["completed_at"] = datetime.now(timezone.utc)

        stmt = (
            update(Session)
            .where(Session.id == session_id)
            .values(status=status, **extras)
        )
        await self.db.execute(stmt)
        # Core UPDATE bypasses ORM identity map; expire cached objects
        # so subsequent get() calls return fresh DB values.
        self.db.expire_all()

    async def update_metrics(
        self,
        session_id: uuid.UUID,
        *,
        total_turns: int,
        total_jailbreaks: int,
        total_borderline: int,
        total_refused: int,
        total_blocked: int,
        total_attacker_tokens: int,
        total_target_tokens: int,
        total_analyzer_tokens: int,
        total_defender_tokens: int,
        estimated_cost_usd: float,
    ) -> None:
        asr = total_jailbreaks / total_turns if total_turns > 0 else 0.0
        stmt = (
            update(Session)
            .where(Session.id == session_id)
            .values(
                total_turns=total_turns,
                total_jailbreaks=total_jailbreaks,
                total_borderline=total_borderline,
                total_refused=total_refused,
                total_blocked=total_blocked,
                asr=asr,
                total_attacker_tokens=total_attacker_tokens,
                total_target_tokens=total_target_tokens,
                total_analyzer_tokens=total_analyzer_tokens,
                total_defender_tokens=total_defender_tokens,
                estimated_cost_usd=estimated_cost_usd,
            )
        )
        await self.db.execute(stmt)

    async def delete(self, session_id: uuid.UUID) -> bool:
        sess = await self.get(session_id)
        if sess is None:
            return False
        await self.db.delete(sess)
        await self.db.flush()
        return True
