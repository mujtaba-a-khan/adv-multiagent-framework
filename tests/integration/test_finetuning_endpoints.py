"""Integration tests for fine-tuning API endpoints using SQLite in-memory DB."""

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

# ── Test DB setup ────────────────────────────────────────────────────────────

TEST_DB_URL = "sqlite+aiosqlite:///file:ft_test?mode=memory&cache=shared&uri=true"

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


# ── Fixtures ─────────────────────────────────────────────────────────────────


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


# ── Helper ───────────────────────────────────────────────────────────────────

VALID_JOB = {
    "name": "Test Pull Job",
    "job_type": "pull_abliterated",
    "source_model": "huihui_ai/qwen3-abliterated:8b",
    "output_model_name": "qwen3-abl",
    "config": {},
}


async def create_job(client: AsyncClient, **overrides: object) -> dict:
    payload = {**VALID_JOB, **overrides}
    resp = await client.post("/api/v1/finetuning/jobs", json=payload)
    assert resp.status_code == 201
    return resp.json()


# ── Tests ────────────────────────────────────────────────────────────────────


class TestCreateJob:
    async def test_create_pull_job(self, client: AsyncClient):
        resp = await client.post("/api/v1/finetuning/jobs", json=VALID_JOB)
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Test Pull Job"
        assert data["job_type"] == "pull_abliterated"
        assert data["status"] == "pending"
        assert data["progress_pct"] == 0.0
        assert "id" in data

    async def test_create_abliterate_job(self, client: AsyncClient):
        data = await create_job(
            client,
            name="Abliterate Llama",
            job_type="abliterate",
            source_model="meta-llama/Llama-3-8B-Instruct",
            output_model_name="llama3-abl",
        )
        assert data["job_type"] == "abliterate"

    async def test_create_sft_job(self, client: AsyncClient):
        data = await create_job(
            client,
            name="SFT Qwen",
            job_type="sft",
            source_model="qwen2:1b",
            output_model_name="qwen2-sft",
            config={"dataset_path": "/tmp/data.jsonl", "lora_rank": 16},
        )
        assert data["job_type"] == "sft"
        assert data["config"]["lora_rank"] == 16

    async def test_create_invalid_job_type(self, client: AsyncClient):
        resp = await client.post(
            "/api/v1/finetuning/jobs",
            json={
                "name": "Bad",
                "job_type": "invalid_type",
                "source_model": "m",
                "output_model_name": "o",
            },
        )
        assert resp.status_code == 422

    async def test_create_missing_required_fields(self, client: AsyncClient):
        resp = await client.post(
            "/api/v1/finetuning/jobs",
            json={"name": "Incomplete"},
        )
        assert resp.status_code == 422


class TestListJobs:
    async def test_list_empty(self, client: AsyncClient):
        resp = await client.get("/api/v1/finetuning/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["jobs"] == []
        assert data["total"] == 0

    async def test_list_multiple(self, client: AsyncClient):
        await create_job(client, name="Job 1")
        await create_job(client, name="Job 2")
        await create_job(client, name="Job 3")

        resp = await client.get("/api/v1/finetuning/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        assert len(data["jobs"]) == 3

    async def test_list_with_pagination(self, client: AsyncClient):
        for i in range(5):
            await create_job(client, name=f"Job {i}")

        resp = await client.get("/api/v1/finetuning/jobs?offset=0&limit=2")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["jobs"]) == 2
        assert data["total"] == 5


class TestGetJob:
    async def test_get_existing(self, client: AsyncClient):
        created = await create_job(client)
        job_id = created["id"]

        resp = await client.get(f"/api/v1/finetuning/jobs/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == job_id
        assert data["name"] == "Test Pull Job"

    async def test_get_not_found(self, client: AsyncClient):
        fake_id = str(uuid.uuid4())
        resp = await client.get(f"/api/v1/finetuning/jobs/{fake_id}")
        assert resp.status_code == 404


class TestDeleteJob:
    async def test_delete_pending_job(self, client: AsyncClient):
        created = await create_job(client)
        job_id = created["id"]

        resp = await client.delete(f"/api/v1/finetuning/jobs/{job_id}")
        assert resp.status_code == 204

        # Verify it's gone
        resp = await client.get(f"/api/v1/finetuning/jobs/{job_id}")
        assert resp.status_code == 404

    async def test_delete_not_found(self, client: AsyncClient):
        fake_id = str(uuid.uuid4())
        resp = await client.delete(f"/api/v1/finetuning/jobs/{fake_id}")
        assert resp.status_code == 404


class TestStartJob:
    async def test_start_not_found(self, client: AsyncClient):
        fake_id = str(uuid.uuid4())
        resp = await client.post(f"/api/v1/finetuning/jobs/{fake_id}/start")
        assert resp.status_code == 404


class TestCancelJob:
    async def test_cancel_not_found(self, client: AsyncClient):
        fake_id = str(uuid.uuid4())
        resp = await client.post(f"/api/v1/finetuning/jobs/{fake_id}/cancel")
        assert resp.status_code == 404

    async def test_cancel_pending_job_fails(self, client: AsyncClient):
        """Cannot cancel a job that hasn't started."""
        created = await create_job(client)
        job_id = created["id"]

        resp = await client.post(f"/api/v1/finetuning/jobs/{job_id}/cancel")
        assert resp.status_code == 409


class TestResponseSchema:
    """Validate the response schema matches expected fields."""

    async def test_job_response_fields(self, client: AsyncClient):
        data = await create_job(client)

        expected_fields = {
            "id",
            "name",
            "job_type",
            "source_model",
            "output_model_name",
            "config",
            "status",
            "progress_pct",
            "current_step",
            "logs",
            "error_message",
            "peak_memory_gb",
            "total_duration_seconds",
            "started_at",
            "completed_at",
            "created_at",
        }
        assert expected_fields.issubset(set(data.keys()))

    async def test_list_response_structure(self, client: AsyncClient):
        await create_job(client)
        resp = await client.get("/api/v1/finetuning/jobs")
        data = resp.json()
        assert "jobs" in data
        assert "total" in data
        assert isinstance(data["jobs"], list)
        assert isinstance(data["total"], int)
