"""Tests for FineTuningJobRepository using in-memory SQLite."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from adversarial_framework.db.models import Base, FineTuningJob
from adversarial_framework.db.repositories.finetuning import (
    FineTuningJobRepository,
)

# ── In-memory SQLite setup ───────────────────────────────────────────────────

TEST_DB_URL = "sqlite+aiosqlite:///file:repo_test?mode=memory&cache=shared&uri=true"

engine = create_async_engine(TEST_DB_URL, echo=False)
TestSessionFactory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture(autouse=True)
async def setup_database():
    """Create all tables before each test, drop after."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def session():
    async with TestSessionFactory() as s:
        yield s


@pytest.fixture
async def repo(session: AsyncSession):
    return FineTuningJobRepository(session)


async def _fresh_get(session: AsyncSession, job_id: uuid.UUID) -> FineTuningJob | None:
    """Bypass the identity-map to get fresh data from SQLite.

    After bulk ``update()`` or ``expire_all()`` the identity-map may hold
    stale / expired objects.  ``session.get()`` can trigger a lazy-load
    that fails under aiosqlite (MissingGreenlet).  ``expunge_all()``
    removes all objects from the identity map so the subsequent
    ``select()`` creates fresh instances from real SQL results.
    """
    session.expunge_all()
    result = await session.execute(select(FineTuningJob).where(FineTuningJob.id == job_id))
    return result.scalar_one_or_none()


# ── Tests ────────────────────────────────────────────────────────────────────


class TestCreate:
    async def test_create_returns_job(self, repo: FineTuningJobRepository):
        job = await repo.create(
            name="Test Pull",
            job_type="pull_abliterated",
            source_model="huihui_ai/qwen3-abliterated:8b",
            output_model_name="qwen3-abl",
            config={"ollama_tag": "huihui_ai/qwen3-abliterated:8b"},
        )
        assert job.id is not None
        assert job.name == "Test Pull"
        assert job.job_type == "pull_abliterated"
        assert job.status == "pending"
        assert job.progress_pct == 0.0

    async def test_create_default_config(self, repo: FineTuningJobRepository):
        job = await repo.create(
            name="Abliterate",
            job_type="abliterate",
            source_model="llama3:8b",
            output_model_name="llama3-abl",
        )
        assert job.config == {}

    async def test_create_with_sft_config(self, repo: FineTuningJobRepository):
        job = await repo.create(
            name="SFT Job",
            job_type="sft",
            source_model="llama3:8b",
            output_model_name="llama3-sft",
            config={
                "dataset_path": "/tmp/data.jsonl",
                "lora_rank": 16,
                "epochs": 3,
            },
        )
        assert job.config["lora_rank"] == 16
        assert job.config["dataset_path"] == "/tmp/data.jsonl"


class TestGet:
    async def test_get_existing(self, repo: FineTuningJobRepository):
        job = await repo.create(
            name="Test",
            job_type="pull_abliterated",
            source_model="model:tag",
            output_model_name="out",
        )
        fetched = await repo.get(job.id)
        assert fetched is not None
        assert fetched.id == job.id
        assert fetched.name == "Test"

    async def test_get_nonexistent(self, repo: FineTuningJobRepository):
        result = await repo.get(uuid.uuid4())
        assert result is None


class TestListAll:
    async def test_list_empty(self, repo: FineTuningJobRepository):
        jobs, total = await repo.list_all()
        assert jobs == []
        assert total == 0

    async def test_list_multiple(self, repo: FineTuningJobRepository):
        for i in range(3):
            await repo.create(
                name=f"Job {i}",
                job_type="pull_abliterated",
                source_model="m",
                output_model_name=f"out-{i}",
            )
        jobs, total = await repo.list_all()
        assert total == 3
        assert len(jobs) == 3

    async def test_list_with_pagination(self, repo: FineTuningJobRepository):
        for i in range(5):
            await repo.create(
                name=f"Job {i}",
                job_type="abliterate",
                source_model="m",
                output_model_name=f"out-{i}",
            )
        jobs, total = await repo.list_all(offset=0, limit=2)
        assert len(jobs) == 2
        assert total == 5

    async def test_list_filter_by_status(self, repo: FineTuningJobRepository):
        await repo.create(
            name="Pending",
            job_type="pull_abliterated",
            source_model="m",
            output_model_name="o1",
        )
        j2 = await repo.create(
            name="Running",
            job_type="abliterate",
            source_model="m",
            output_model_name="o2",
        )
        await repo.update_status(j2.id, "running")

        pending_jobs, pending_total = await repo.list_all(status="pending")
        assert pending_total == 1
        assert pending_jobs[0].name == "Pending"

        running_jobs, running_total = await repo.list_all(status="running")
        assert running_total == 1


class TestUpdateStatus:
    async def test_status_to_running_sets_started_at(
        self, repo: FineTuningJobRepository, session: AsyncSession
    ):
        job = await repo.create(
            name="Test",
            job_type="abliterate",
            source_model="m",
            output_model_name="o",
        )
        job_id = job.id  # capture before expire_all
        await repo.update_status(job_id, "running")

        refreshed = await _fresh_get(session, job_id)
        assert refreshed is not None
        assert refreshed.status == "running"
        assert refreshed.started_at is not None

    async def test_status_to_completed_sets_completed_at(
        self, repo: FineTuningJobRepository, session: AsyncSession
    ):
        job = await repo.create(
            name="Test",
            job_type="abliterate",
            source_model="m",
            output_model_name="o",
        )
        job_id = job.id  # capture before expire_all
        await repo.update_status(job_id, "running")
        await repo.update_status(job_id, "completed")

        refreshed = await _fresh_get(session, job_id)
        assert refreshed is not None
        assert refreshed.status == "completed"
        assert refreshed.completed_at is not None

    async def test_status_to_failed_stores_error(
        self, repo: FineTuningJobRepository, session: AsyncSession
    ):
        job = await repo.create(
            name="Test",
            job_type="sft",
            source_model="m",
            output_model_name="o",
        )
        job_id = job.id  # capture before expire_all
        await repo.update_status(job_id, "failed", error_message="Out of memory")

        refreshed = await _fresh_get(session, job_id)
        assert refreshed is not None
        assert refreshed.status == "failed"
        assert refreshed.error_message == "Out of memory"
        assert refreshed.completed_at is not None

    async def test_status_to_cancelled(self, repo: FineTuningJobRepository, session: AsyncSession):
        job = await repo.create(
            name="Test",
            job_type="pull_abliterated",
            source_model="m",
            output_model_name="o",
        )
        job_id = job.id  # capture before expire_all
        await repo.update_status(job_id, "cancelled")

        refreshed = await _fresh_get(session, job_id)
        assert refreshed is not None
        assert refreshed.status == "cancelled"
        assert refreshed.completed_at is not None


class TestUpdateProgress:
    async def test_update_progress_pct(self, repo: FineTuningJobRepository, session: AsyncSession):
        job = await repo.create(
            name="Test",
            job_type="abliterate",
            source_model="m",
            output_model_name="o",
        )
        await repo.update_progress(job.id, progress_pct=42.5, current_step="Loading model")

        refreshed = await _fresh_get(session, job.id)
        assert refreshed is not None
        assert refreshed.progress_pct == 42.5
        assert refreshed.current_step == "Loading model"

    async def test_update_with_duration(self, repo: FineTuningJobRepository, session: AsyncSession):
        job = await repo.create(
            name="Test",
            job_type="sft",
            source_model="m",
            output_model_name="o",
        )
        await repo.update_progress(
            job.id,
            progress_pct=80.0,
            current_step="Training",
            total_duration_seconds=120.5,
        )

        refreshed = await _fresh_get(session, job.id)
        assert refreshed is not None
        assert refreshed.total_duration_seconds == 120.5

    async def test_update_with_peak_memory(
        self, repo: FineTuningJobRepository, session: AsyncSession
    ):
        job = await repo.create(
            name="Test",
            job_type="abliterate",
            source_model="m",
            output_model_name="o",
        )
        await repo.update_progress(
            job.id,
            progress_pct=55.0,
            current_step="Computing refusal dir",
            peak_memory_gb=7.2,
        )

        refreshed = await _fresh_get(session, job.id)
        assert refreshed is not None
        assert refreshed.peak_memory_gb == 7.2


class TestAppendLog:
    async def test_append_single_log(self, repo: FineTuningJobRepository):
        job = await repo.create(
            name="Test",
            job_type="abliterate",
            source_model="m",
            output_model_name="o",
        )
        await repo.append_log(job.id, {"level": "info", "message": "Started"})

        refreshed = await repo.get(job.id)
        assert refreshed is not None
        assert len(refreshed.logs) == 1
        assert refreshed.logs[0]["message"] == "Started"

    async def test_append_multiple_logs(self, repo: FineTuningJobRepository):
        job = await repo.create(
            name="Test",
            job_type="sft",
            source_model="m",
            output_model_name="o",
        )
        await repo.append_log(job.id, {"level": "info", "message": "Step 1"})
        await repo.append_log(job.id, {"level": "info", "message": "Step 2"})

        refreshed = await repo.get(job.id)
        assert refreshed is not None
        assert len(refreshed.logs) == 2

    async def test_append_log_nonexistent_job(self, repo: FineTuningJobRepository):
        """Appending to a missing job is a no-op."""
        await repo.append_log(uuid.uuid4(), {"level": "info", "message": "ghost"})
        # No exception raised


class TestDelete:
    async def test_delete_existing(self, repo: FineTuningJobRepository):
        job = await repo.create(
            name="To Delete",
            job_type="pull_abliterated",
            source_model="m",
            output_model_name="o",
        )
        result = await repo.delete(job.id)
        assert result is True

        gone = await repo.get(job.id)
        assert gone is None

    async def test_delete_nonexistent(self, repo: FineTuningJobRepository):
        result = await repo.delete(uuid.uuid4())
        assert result is False
