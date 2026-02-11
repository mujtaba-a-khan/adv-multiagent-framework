"""Integration tests for playground API endpoints using SQLite in-memory DB."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from adversarial_framework.api.app import create_app
from adversarial_framework.api.dependencies import get_db
from adversarial_framework.db.models import Base

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

# Test DB setup

TEST_DB_URL = "sqlite+aiosqlite:///file:playground_test?mode=memory&cache=shared&uri=true"

engine = create_async_engine(TEST_DB_URL, echo=False)
TestSessionFactory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
    async with TestSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# Fixtures

@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
async def setup_database():
    """Create all tables before each test, drop after."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# Helper

VALID_CONVERSATION = {
    "title": "Test Chat",
    "target_model": "llama3:8b",
}


async def create_conversation(client: AsyncClient, **overrides: object) -> dict:
    payload = {**VALID_CONVERSATION, **overrides}
    resp = await client.post("/api/v1/playground/conversations", json=payload)
    assert resp.status_code == 201
    return resp.json()


# Tests


class TestCreateConversation:
    async def test_create_basic(self, client: AsyncClient):
        resp = await client.post("/api/v1/playground/conversations", json=VALID_CONVERSATION)
        assert resp.status_code == 201
        data = resp.json()
        assert data["title"] == "Test Chat"
        assert data["target_model"] == "llama3:8b"
        assert data["target_provider"] == "ollama"
        assert data["system_prompt"] is None
        assert data["active_defenses"] == []
        assert data["total_messages"] == 0
        assert data["total_jailbreaks"] == 0
        assert data["total_blocked"] == 0
        assert "id" in data

    async def test_create_with_system_prompt(self, client: AsyncClient):
        data = await create_conversation(client, system_prompt="You are a helpful assistant.")
        assert data["system_prompt"] == "You are a helpful assistant."

    async def test_create_with_defenses(self, client: AsyncClient):
        data = await create_conversation(
            client,
            active_defenses=[{"name": "rule_based"}],
        )
        assert len(data["active_defenses"]) == 1
        assert data["active_defenses"][0]["name"] == "rule_based"

    async def test_create_missing_title(self, client: AsyncClient):
        resp = await client.post(
            "/api/v1/playground/conversations",
            json={"target_model": "llama3:8b"},
        )
        assert resp.status_code == 422

    async def test_create_missing_target_model(self, client: AsyncClient):
        resp = await client.post(
            "/api/v1/playground/conversations",
            json={"title": "No Model"},
        )
        assert resp.status_code == 422


class TestListConversations:
    async def test_list_empty(self, client: AsyncClient):
        resp = await client.get("/api/v1/playground/conversations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["conversations"] == []
        assert data["total"] == 0

    async def test_list_multiple(self, client: AsyncClient):
        await create_conversation(client, title="Chat 1")
        await create_conversation(client, title="Chat 2")
        await create_conversation(client, title="Chat 3")

        resp = await client.get("/api/v1/playground/conversations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        assert len(data["conversations"]) == 3

    async def test_list_pagination(self, client: AsyncClient):
        for i in range(5):
            await create_conversation(client, title=f"Chat {i}")

        resp = await client.get("/api/v1/playground/conversations?offset=0&limit=2")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["conversations"]) == 2
        assert data["total"] == 5


class TestGetConversation:
    async def test_get_existing(self, client: AsyncClient):
        created = await create_conversation(client)
        conv_id = created["id"]

        resp = await client.get(f"/api/v1/playground/conversations/{conv_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == conv_id
        assert data["title"] == "Test Chat"

    async def test_get_not_found(self, client: AsyncClient):
        fake_id = str(uuid.uuid4())
        resp = await client.get(f"/api/v1/playground/conversations/{fake_id}")
        assert resp.status_code == 404


class TestUpdateConversation:
    async def test_update_title(self, client: AsyncClient):
        created = await create_conversation(client)
        conv_id = created["id"]

        resp = await client.patch(
            f"/api/v1/playground/conversations/{conv_id}",
            json={"title": "Updated Title"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == "Updated Title"

    async def test_update_system_prompt(self, client: AsyncClient):
        created = await create_conversation(client)
        conv_id = created["id"]

        resp = await client.patch(
            f"/api/v1/playground/conversations/{conv_id}",
            json={"system_prompt": "New system prompt"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["system_prompt"] == "New system prompt"

    async def test_update_defenses(self, client: AsyncClient):
        created = await create_conversation(client)
        conv_id = created["id"]

        resp = await client.patch(
            f"/api/v1/playground/conversations/{conv_id}",
            json={
                "active_defenses": [
                    {"name": "rule_based", "params": {}},
                ]
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["active_defenses"]) == 1
        assert data["active_defenses"][0]["name"] == "rule_based"

    async def test_update_no_fields(self, client: AsyncClient):
        created = await create_conversation(client)
        conv_id = created["id"]

        resp = await client.patch(
            f"/api/v1/playground/conversations/{conv_id}",
            json={},
        )
        assert resp.status_code == 400

    async def test_update_not_found(self, client: AsyncClient):
        fake_id = str(uuid.uuid4())
        resp = await client.patch(
            f"/api/v1/playground/conversations/{fake_id}",
            json={"title": "Ghost"},
        )
        assert resp.status_code == 404


class TestDeleteConversation:
    async def test_delete_existing(self, client: AsyncClient):
        created = await create_conversation(client)
        conv_id = created["id"]

        resp = await client.delete(f"/api/v1/playground/conversations/{conv_id}")
        assert resp.status_code == 204

        # Verify it's gone
        resp = await client.get(f"/api/v1/playground/conversations/{conv_id}")
        assert resp.status_code == 404

    async def test_delete_not_found(self, client: AsyncClient):
        fake_id = str(uuid.uuid4())
        resp = await client.delete(f"/api/v1/playground/conversations/{fake_id}")
        assert resp.status_code == 404


class TestListMessages:
    async def test_list_empty_messages(self, client: AsyncClient):
        created = await create_conversation(client)
        conv_id = created["id"]

        resp = await client.get(f"/api/v1/playground/conversations/{conv_id}/messages")
        assert resp.status_code == 200
        data = resp.json()
        assert data["messages"] == []
        assert data["total"] == 0

    async def test_list_messages_not_found(self, client: AsyncClient):
        fake_id = str(uuid.uuid4())
        resp = await client.get(f"/api/v1/playground/conversations/{fake_id}/messages")
        assert resp.status_code == 404


class TestSendMessage:
    async def test_send_not_found(self, client: AsyncClient):
        fake_id = str(uuid.uuid4())
        resp = await client.post(
            f"/api/v1/playground/conversations/{fake_id}/messages",
            json={"prompt": "Hello"},
        )
        assert resp.status_code == 404


class TestResponseSchema:
    """Validate the response schema matches expected fields."""

    async def test_conversation_response_fields(self, client: AsyncClient):
        data = await create_conversation(client)

        expected_fields = {
            "id",
            "title",
            "target_model",
            "target_provider",
            "system_prompt",
            "analyzer_model",
            "active_defenses",
            "total_messages",
            "total_jailbreaks",
            "total_blocked",
            "total_target_tokens",
            "total_analyzer_tokens",
            "created_at",
            "updated_at",
        }
        assert expected_fields.issubset(set(data.keys()))

    async def test_list_response_structure(self, client: AsyncClient):
        await create_conversation(client)
        resp = await client.get("/api/v1/playground/conversations")
        data = resp.json()
        assert "conversations" in data
        assert "total" in data
        assert isinstance(data["conversations"], list)
        assert isinstance(data["total"], int)

    async def test_messages_list_response_structure(self, client: AsyncClient):
        created = await create_conversation(client)
        conv_id = created["id"]

        resp = await client.get(f"/api/v1/playground/conversations/{conv_id}/messages")
        data = resp.json()
        assert "messages" in data
        assert "total" in data
        assert isinstance(data["messages"], list)
        assert isinstance(data["total"], int)
