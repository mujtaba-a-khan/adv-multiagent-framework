"""Repository for Playground CRUD operations."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from adversarial_framework.db.models import (
    PlaygroundConversation,
    PlaygroundMessage,
)


class PlaygroundRepository:
    """Async CRUD for playground conversations and messages."""

    def __init__(self, session: AsyncSession) -> None:
        self.db = session

    # Conversation CRUD

    async def create_conversation(
        self, **kwargs: Any
    ) -> PlaygroundConversation:
        conversation = PlaygroundConversation(**kwargs)
        self.db.add(conversation)
        await self.db.flush()
        return conversation

    async def get_conversation(
        self, conversation_id: uuid.UUID
    ) -> PlaygroundConversation | None:
        return await self.db.get(
            PlaygroundConversation, conversation_id
        )

    async def list_conversations(
        self, offset: int = 0, limit: int = 50
    ) -> tuple[Sequence[PlaygroundConversation], int]:
        stmt = (
            select(PlaygroundConversation)
            .order_by(PlaygroundConversation.updated_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        conversations = result.scalars().all()

        count_result = await self.db.execute(
            select(func.count()).select_from(
                PlaygroundConversation
            )
        )
        total = count_result.scalar_one()
        return conversations, total

    async def update_conversation(
        self, conversation_id: uuid.UUID, **kwargs: Any
    ) -> PlaygroundConversation | None:
        conversation = await self.get_conversation(conversation_id)
        if conversation is None:
            return None
        for key, value in kwargs.items():
            setattr(conversation, key, value)
        await self.db.flush()
        await self.db.refresh(conversation)
        return conversation

    async def delete_conversation(
        self, conversation_id: uuid.UUID
    ) -> bool:
        conversation = await self.get_conversation(conversation_id)
        if conversation is None:
            return False
        await self.db.delete(conversation)
        await self.db.flush()
        return True

    # Message CRUD

    async def create_message(
        self, conversation_id: uuid.UUID, **kwargs: Any
    ) -> PlaygroundMessage:
        message = PlaygroundMessage(
            conversation_id=conversation_id, **kwargs
        )
        self.db.add(message)
        await self.db.flush()
        return message

    async def list_messages(
        self,
        conversation_id: uuid.UUID,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[Sequence[PlaygroundMessage], int]:
        stmt = (
            select(PlaygroundMessage)
            .where(
                PlaygroundMessage.conversation_id
                == conversation_id
            )
            .order_by(PlaygroundMessage.message_number.asc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        messages = result.scalars().all()

        count_result = await self.db.execute(
            select(func.count())
            .select_from(PlaygroundMessage)
            .where(
                PlaygroundMessage.conversation_id
                == conversation_id
            )
        )
        total = count_result.scalar_one()
        return messages, total
