"""Repository for AbliterationPrompt CRUD operations."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from adversarial_framework.db.models import AbliterationPrompt


class AbliterationPromptRepository:
    """Async CRUD for abliteration dataset prompts."""

    def __init__(self, session: AsyncSession) -> None:
        self.db = session

    async def create(self, **kwargs: object) -> AbliterationPrompt:
        prompt = AbliterationPrompt(**kwargs)
        self.db.add(prompt)
        await self.db.flush()
        return prompt

    async def create_many(self, prompts: list[dict[str, object]]) -> list[AbliterationPrompt]:
        objects = [AbliterationPrompt(**p) for p in prompts]
        self.db.add_all(objects)
        await self.db.flush()
        return objects

    async def get(self, prompt_id: uuid.UUID) -> AbliterationPrompt | None:
        return await self.db.get(AbliterationPrompt, prompt_id)

    async def list_all(
        self,
        offset: int = 0,
        limit: int = 50,
        category: str | None = None,
        source: str | None = None,
    ) -> tuple[Sequence[AbliterationPrompt], int]:
        stmt = (
            select(AbliterationPrompt)
            .where(AbliterationPrompt.status == "active")
            .order_by(AbliterationPrompt.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        count_stmt = (
            select(func.count())
            .select_from(AbliterationPrompt)
            .where(AbliterationPrompt.status == "active")
        )

        if category is not None:
            stmt = stmt.where(AbliterationPrompt.category == category)
            count_stmt = count_stmt.where(AbliterationPrompt.category == category)
        if source is not None:
            stmt = stmt.where(AbliterationPrompt.source == source)
            count_stmt = count_stmt.where(AbliterationPrompt.source == source)

        result = await self.db.execute(stmt)
        prompts = result.scalars().all()

        count_result = await self.db.execute(count_stmt)
        total = count_result.scalar_one()

        return prompts, total

    async def list_by_category(self, category: str) -> Sequence[AbliterationPrompt]:
        """Return ALL active prompts of a category (used by pipeline)."""
        stmt = (
            select(AbliterationPrompt)
            .where(AbliterationPrompt.category == category)
            .where(AbliterationPrompt.status == "active")
            .order_by(AbliterationPrompt.created_at.asc())
        )
        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def count_by_category(self) -> dict[str, int]:
        """Return counts of active prompts per category."""
        stmt = (
            select(
                AbliterationPrompt.category,
                func.count(),
            )
            .where(AbliterationPrompt.status == "active")
            .group_by(AbliterationPrompt.category)
        )
        result = await self.db.execute(stmt)
        counts: dict[str, int] = {"harmful": 0, "harmless": 0}
        for category, count in result.all():
            counts[category] = count
        return counts

    async def list_suggestions(
        self,
    ) -> tuple[Sequence[AbliterationPrompt], int]:
        """Return all suggested (pending review) prompts."""
        stmt = (
            select(AbliterationPrompt)
            .where(AbliterationPrompt.status == "suggested")
            .order_by(AbliterationPrompt.created_at.desc())
        )
        result = await self.db.execute(stmt)
        prompts = result.scalars().all()

        count_stmt = (
            select(func.count())
            .select_from(AbliterationPrompt)
            .where(AbliterationPrompt.status == "suggested")
        )
        count_result = await self.db.execute(count_stmt)
        total = count_result.scalar_one()

        return prompts, total

    async def confirm(self, prompt_id: uuid.UUID) -> AbliterationPrompt | None:
        """Change status from 'suggested' to 'active'."""
        await self.db.execute(
            update(AbliterationPrompt)
            .where(AbliterationPrompt.id == prompt_id)
            .values(status="active")
        )
        return await self.get(prompt_id)

    async def exists_by_text(self, text: str) -> bool:
        """Check if a prompt with this exact text already exists."""
        stmt = (
            select(func.count())
            .select_from(AbliterationPrompt)
            .where(AbliterationPrompt.text == text)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one() > 0

    _UNSET: object = object()

    async def update(
        self,
        prompt_id: uuid.UUID,
        *,
        text: str | None = None,
        category: str | None = None,
        pair_id: object = _UNSET,
    ) -> AbliterationPrompt | None:
        values: dict[str, object] = {}
        if text is not None:
            values["text"] = text
        if category is not None:
            values["category"] = category
        if pair_id is not self._UNSET:
            values["pair_id"] = pair_id
        if not values:
            return await self.get(prompt_id)

        await self.db.execute(
            update(AbliterationPrompt).where(AbliterationPrompt.id == prompt_id).values(**values)
        )
        return await self.get(prompt_id)

    async def delete(self, prompt_id: uuid.UUID) -> bool:
        prompt = await self.get(prompt_id)
        if prompt is None:
            return False
        await self.db.delete(prompt)
        await self.db.flush()
        return True

    async def delete_all(self) -> int:
        result = await self.db.execute(delete(AbliterationPrompt))
        return result.rowcount  # type: ignore[attr-defined, no-any-return]
