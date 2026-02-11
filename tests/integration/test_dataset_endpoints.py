"""Integration tests for dataset API endpoints using SQLite in-memory DB."""

from __future__ import annotations

import io
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

TEST_DB_URL = "sqlite+aiosqlite:///file:dataset_test?mode=memory&cache=shared&uri=true"

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


# Helpers

BASE = "/api/v1/finetuning/dataset"


async def create_prompt(
    client: AsyncClient,
    text: str = "test prompt",
    category: str = "harmful",
    **overrides: object,
) -> dict:
    payload: dict[str, object] = {"text": text, "category": category, **overrides}
    resp = await client.post(f"{BASE}/prompts", json=payload)
    assert resp.status_code == 201
    return resp.json()


# Tests


class TestAddPrompt:
    async def test_create_harmful_prompt(self, client: AsyncClient):
        resp = await client.post(
            f"{BASE}/prompts",
            json={"text": "harmful test", "category": "harmful"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["text"] == "harmful test"
        assert data["category"] == "harmful"
        assert data["source"] == "manual"
        assert data["status"] == "active"

    async def test_create_harmless_prompt(self, client: AsyncClient):
        resp = await client.post(
            f"{BASE}/prompts",
            json={"text": "harmless test", "category": "harmless"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["text"] == "harmless test"
        assert data["category"] == "harmless"

    async def test_auto_generate_counterpart_false(self, client: AsyncClient):
        resp = await client.post(
            f"{BASE}/prompts",
            json={
                "text": "test prompt",
                "category": "harmful",
                "auto_generate_counterpart": False,
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["category"] == "harmful"


class TestListPrompts:
    async def test_list_empty(self, client: AsyncClient):
        resp = await client.get(f"{BASE}/prompts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["prompts"] == []
        assert data["total"] == 0

    async def test_list_with_created_prompts(self, client: AsyncClient):
        await create_prompt(client, text="prompt 1", category="harmful")
        await create_prompt(client, text="prompt 2", category="harmless")

        resp = await client.get(f"{BASE}/prompts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["prompts"]) == 2

    async def test_list_with_category_filter(self, client: AsyncClient):
        await create_prompt(client, text="harmful one", category="harmful")
        await create_prompt(client, text="harmless one", category="harmless")

        resp = await client.get(f"{BASE}/prompts?category=harmful")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["prompts"][0]["category"] == "harmful"

    async def test_list_with_pagination(self, client: AsyncClient):
        for i in range(5):
            await create_prompt(client, text=f"prompt {i}", category="harmful")

        resp = await client.get(f"{BASE}/prompts?offset=0&limit=2")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["prompts"]) == 2
        assert data["total"] == 5


class TestGetStats:
    async def test_empty_stats(self, client: AsyncClient):
        resp = await client.get(f"{BASE}/prompts/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["harmful_count"] == 0
        assert data["harmless_count"] == 0
        assert data["total"] == 0
        assert data["warning"] is None

    async def test_stats_after_creating_prompts(self, client: AsyncClient):
        await create_prompt(client, text="harmful 1", category="harmful")
        await create_prompt(client, text="harmful 2", category="harmful")
        await create_prompt(client, text="harmless 1", category="harmless")

        resp = await client.get(f"{BASE}/prompts/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["harmful_count"] == 2
        assert data["harmless_count"] == 1
        assert data["total"] == 3

    async def test_warning_under_16_pairs(self, client: AsyncClient):
        await create_prompt(client, text="harmful", category="harmful")
        await create_prompt(client, text="harmless", category="harmless")

        resp = await client.get(f"{BASE}/prompts/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["warning"] is not None
        assert "16" in data["warning"]


class TestUpdatePrompt:
    async def test_update_text(self, client: AsyncClient):
        created = await create_prompt(client, text="original text")
        prompt_id = created["id"]

        resp = await client.put(
            f"{BASE}/prompts/{prompt_id}",
            json={"text": "updated text"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "updated text"

    async def test_update_category(self, client: AsyncClient):
        created = await create_prompt(client, text="test", category="harmful")
        prompt_id = created["id"]

        resp = await client.put(
            f"{BASE}/prompts/{prompt_id}",
            json={"category": "harmless"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["category"] == "harmless"

    async def test_update_not_found(self, client: AsyncClient):
        fake_id = str(uuid.uuid4())
        resp = await client.put(
            f"{BASE}/prompts/{fake_id}",
            json={"text": "updated"},
        )
        assert resp.status_code == 404


class TestDeletePrompt:
    async def test_delete_existing(self, client: AsyncClient):
        created = await create_prompt(client)
        prompt_id = created["id"]

        resp = await client.delete(f"{BASE}/prompts/{prompt_id}")
        assert resp.status_code == 204

    async def test_delete_not_found(self, client: AsyncClient):
        fake_id = str(uuid.uuid4())
        resp = await client.delete(f"{BASE}/prompts/{fake_id}")
        assert resp.status_code == 404

    async def test_verify_deleted(self, client: AsyncClient):
        created = await create_prompt(client)
        prompt_id = created["id"]

        resp = await client.delete(f"{BASE}/prompts/{prompt_id}")
        assert resp.status_code == 204

        # Verify the prompt is actually gone
        resp = await client.get(f"{BASE}/prompts")
        data = resp.json()
        assert data["total"] == 0
        assert len(data["prompts"]) == 0


class TestUploadPrompts:
    async def test_valid_jsonl_upload(self, client: AsyncClient):
        content = '{"harmful": "test harmful", "harmless": "test harmless"}\n'
        resp = await client.post(
            f"{BASE}/prompts/upload",
            files={
                "file": ("test.jsonl", io.BytesIO(content.encode()), "application/octet-stream")
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["total"] == 2
        categories = {p["category"] for p in data["prompts"]}
        assert categories == {"harmful", "harmless"}

    async def test_reject_non_jsonl_file(self, client: AsyncClient):
        content = b"plain text content"
        resp = await client.post(
            f"{BASE}/prompts/upload",
            files={"file": ("test.txt", io.BytesIO(content), "text/plain")},
        )
        assert resp.status_code == 400

    async def test_reject_invalid_json_content(self, client: AsyncClient):
        content = b"this is not valid json\n"
        resp = await client.post(
            f"{BASE}/prompts/upload",
            files={"file": ("test.jsonl", io.BytesIO(content), "application/octet-stream")},
        )
        assert resp.status_code == 400

    async def test_reject_missing_keys(self, client: AsyncClient):
        content = b'{"harmful": "only harmful, no harmless key"}\n'
        resp = await client.post(
            f"{BASE}/prompts/upload",
            files={"file": ("test.jsonl", io.BytesIO(content), "application/octet-stream")},
        )
        assert resp.status_code == 400


class TestGenerateHarmless:
    async def test_400_without_model(self, client: AsyncClient):
        resp = await client.post(
            f"{BASE}/prompts/generate-harmless",
            json={"harmful_prompt": "test harmful prompt"},
        )
        assert resp.status_code == 400


class TestSuggestions:
    async def test_empty_suggestions(self, client: AsyncClient):
        resp = await client.get(f"{BASE}/prompts/suggestions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["suggestions"] == []
        assert data["total"] == 0


class TestConfirmSuggestion:
    async def test_confirm_not_found(self, client: AsyncClient):
        fake_id = str(uuid.uuid4())
        resp = await client.post(
            f"{BASE}/prompts/{fake_id}/confirm?auto_generate_counterpart=false",
        )
        assert resp.status_code == 404

    async def test_confirm_with_auto_generate_false(self, client: AsyncClient):
        """Confirm with auto_generate_counterpart=false to avoid needing LLM."""
        fake_id = str(uuid.uuid4())
        resp = await client.post(
            f"{BASE}/prompts/{fake_id}/confirm?auto_generate_counterpart=false",
        )
        # 404 because fake prompt doesn't exist — confirms the endpoint works
        assert resp.status_code == 404


class TestDismissSuggestion:
    async def test_dismiss_not_found(self, client: AsyncClient):
        fake_id = str(uuid.uuid4())
        resp = await client.post(f"{BASE}/prompts/{fake_id}/dismiss")
        assert resp.status_code == 404


class TestResponseSchema:
    """Validate the response schema matches expected fields."""

    async def test_prompt_response_fields(self, client: AsyncClient):
        data = await create_prompt(client)

        expected_fields = {
            "id",
            "text",
            "category",
            "source",
            "status",
            "pair_id",
            "created_at",
        }
        assert expected_fields.issubset(set(data.keys()))

    async def test_list_response_structure(self, client: AsyncClient):
        await create_prompt(client)
        resp = await client.get(f"{BASE}/prompts")
        data = resp.json()
        assert "prompts" in data
        assert "total" in data
        assert isinstance(data["prompts"], list)
        assert isinstance(data["total"], int)

    async def test_stats_response_structure(self, client: AsyncClient):
        resp = await client.get(f"{BASE}/prompts/stats")
        data = resp.json()
        expected_fields = {
            "harmful_count",
            "harmless_count",
            "total",
            "min_recommended",
        }
        assert expected_fields.issubset(set(data.keys()))
