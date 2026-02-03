"""Integration tests for FastAPI endpoints using SQLite in-memory DB.

These tests override the DB dependency to use an async SQLite engine so
they can run without a real PostgreSQL / Neon connection.
"""

from __future__ import annotations

import uuid
from typing import AsyncGenerator

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

# ── Test DB setup ────────────────────────────────────────────────────────────

TEST_DB_URL = "sqlite+aiosqlite:///file::memory:?cache=shared&uri=true"

engine = create_async_engine(TEST_DB_URL, echo=False)
TestSessionFactory = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)


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
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ── Tests ────────────────────────────────────────────────────────────────────


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health(self, client: AsyncClient):
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestExperimentEndpoints:
    @pytest.mark.asyncio
    async def test_create_experiment(self, client: AsyncClient):
        payload = {
            "name": "Test Experiment",
            "target_model": "llama3:8b",
            "attacker_model": "phi4-reasoning:14b",
            "analyzer_model": "phi4-reasoning:14b",
            "defender_model": "qwen3:8b",
            "attack_objective": "Test objective for safety research",
            "strategy_name": "pair",
        }
        resp = await client.post("/api/v1/experiments", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Test Experiment"
        assert data["target_model"] == "llama3:8b"
        assert data["strategy_name"] == "pair"
        assert "id" in data

    @pytest.mark.asyncio
    async def test_list_experiments(self, client: AsyncClient):
        # Create two experiments
        for i in range(2):
            await client.post("/api/v1/experiments", json={
                "name": f"Exp {i}",
                "target_model": "llama3:8b",
                "attacker_model": "phi4-reasoning:14b",
                "analyzer_model": "phi4-reasoning:14b",
                "defender_model": "qwen3:8b",
                "attack_objective": "Test",
                "strategy_name": "pair",
            })
        resp = await client.get("/api/v1/experiments")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 2

    @pytest.mark.asyncio
    async def test_get_experiment(self, client: AsyncClient):
        create_resp = await client.post("/api/v1/experiments", json={
            "name": "Get Test",
            "target_model": "llama3:8b",
            "attacker_model": "phi4-reasoning:14b",
            "analyzer_model": "phi4-reasoning:14b",
            "defender_model": "qwen3:8b",
            "attack_objective": "Test",
            "strategy_name": "pair",
        })
        exp_id = create_resp.json()["id"]
        resp = await client.get(f"/api/v1/experiments/{exp_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "Get Test"

    @pytest.mark.asyncio
    async def test_get_experiment_not_found(self, client: AsyncClient):
        fake_id = str(uuid.uuid4())
        resp = await client.get(f"/api/v1/experiments/{fake_id}")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_update_experiment(self, client: AsyncClient):
        create_resp = await client.post("/api/v1/experiments", json={
            "name": "Update Test",
            "target_model": "llama3:8b",
            "attacker_model": "phi4-reasoning:14b",
            "analyzer_model": "phi4-reasoning:14b",
            "defender_model": "qwen3:8b",
            "attack_objective": "Test",
            "strategy_name": "pair",
        })
        exp_id = create_resp.json()["id"]
        resp = await client.patch(
            f"/api/v1/experiments/{exp_id}",
            json={"name": "Updated Name"},
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "Updated Name"

    @pytest.mark.asyncio
    async def test_delete_experiment(self, client: AsyncClient):
        create_resp = await client.post("/api/v1/experiments", json={
            "name": "Delete Test",
            "target_model": "llama3:8b",
            "attacker_model": "phi4-reasoning:14b",
            "analyzer_model": "phi4-reasoning:14b",
            "defender_model": "qwen3:8b",
            "attack_objective": "Test",
            "strategy_name": "pair",
        })
        exp_id = create_resp.json()["id"]
        resp = await client.delete(f"/api/v1/experiments/{exp_id}")
        assert resp.status_code == 204

        # Verify deleted
        resp = await client.get(f"/api/v1/experiments/{exp_id}")
        assert resp.status_code == 404


class TestSessionEndpoints:
    @pytest.mark.asyncio
    async def test_create_session(self, client: AsyncClient):
        # Create experiment first
        exp_resp = await client.post("/api/v1/experiments", json={
            "name": "Session Test",
            "target_model": "llama3:8b",
            "attacker_model": "phi4-reasoning:14b",
            "analyzer_model": "phi4-reasoning:14b",
            "defender_model": "qwen3:8b",
            "attack_objective": "Test",
            "strategy_name": "pair",
        })
        exp_id = exp_resp.json()["id"]

        resp = await client.post(f"/api/v1/experiments/{exp_id}/sessions")
        assert resp.status_code == 201
        data = resp.json()
        assert data["experiment_id"] == exp_id
        assert data["status"] == "pending"

    @pytest.mark.asyncio
    async def test_list_sessions(self, client: AsyncClient):
        exp_resp = await client.post("/api/v1/experiments", json={
            "name": "List Sessions Test",
            "target_model": "llama3:8b",
            "attacker_model": "phi4-reasoning:14b",
            "analyzer_model": "phi4-reasoning:14b",
            "defender_model": "qwen3:8b",
            "attack_objective": "Test",
            "strategy_name": "pair",
        })
        exp_id = exp_resp.json()["id"]

        # Create two sessions
        await client.post(f"/api/v1/experiments/{exp_id}/sessions")
        await client.post(f"/api/v1/experiments/{exp_id}/sessions")

        resp = await client.get(f"/api/v1/experiments/{exp_id}/sessions")
        assert resp.status_code == 200
        assert resp.json()["total"] >= 2


class TestStrategyEndpoints:
    @pytest.mark.asyncio
    async def test_list_strategies(self, client: AsyncClient):
        resp = await client.get("/api/v1/strategies")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        names = [s["name"] for s in data["strategies"]]
        assert "pair" in names

    @pytest.mark.asyncio
    async def test_get_strategy(self, client: AsyncClient):
        resp = await client.get("/api/v1/strategies/pair")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "pair"
        assert data["category"] == "optimization"

    @pytest.mark.asyncio
    async def test_get_strategy_not_found(self, client: AsyncClient):
        resp = await client.get("/api/v1/strategies/nonexistent_xyz")
        assert resp.status_code == 404
