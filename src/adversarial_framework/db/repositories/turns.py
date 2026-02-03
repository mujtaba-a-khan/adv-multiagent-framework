"""Repository for Turn CRUD operations."""

from __future__ import annotations

import uuid
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from adversarial_framework.db.models import Turn


class TurnRepository:
    """Async CRUD for attack turns."""

    def __init__(self, session: AsyncSession) -> None:
        self.db = session

    async def create(self, session_id: uuid.UUID, **kwargs: object) -> Turn:
        turn = Turn(session_id=session_id, **kwargs)
        self.db.add(turn)
        await self.db.flush()
        return turn

    async def get(self, turn_id: uuid.UUID) -> Turn | None:
        return await self.db.get(Turn, turn_id)

    async def list_by_session(
        self, session_id: uuid.UUID, offset: int = 0, limit: int = 100
    ) -> Sequence[Turn]:
        stmt = (
            select(Turn)
            .where(Turn.session_id == session_id)
            .order_by(Turn.turn_number.asc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def get_latest(self, session_id: uuid.UUID) -> Turn | None:
        stmt = (
            select(Turn)
            .where(Turn.session_id == session_id)
            .order_by(Turn.turn_number.desc())
            .limit(1)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def count_by_verdict(
        self, session_id: uuid.UUID, verdict: str
    ) -> int:
        stmt = (
            select(Turn)
            .where(Turn.session_id == session_id, Turn.judge_verdict == verdict)
        )
        result = await self.db.execute(stmt)
        return len(result.scalars().all())
