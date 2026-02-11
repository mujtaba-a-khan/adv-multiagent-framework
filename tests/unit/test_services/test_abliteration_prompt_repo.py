"""Tests for AbliterationPromptRepository using in-memory SQLite."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from adversarial_framework.db.models import Base
from adversarial_framework.db.repositories.abliteration_prompts import (
    AbliterationPromptRepository,
)

# ── In-memory SQLite setup ───────────────────────────────────────────────────

TEST_DB_URL = "sqlite+aiosqlite:///file:abl_repo_test?mode=memory&cache=shared&uri=true"

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
    return AbliterationPromptRepository(session)


# ── Tests ────────────────────────────────────────────────────────────────────


class TestCreate:
    async def test_create_returns_prompt(self, repo: AbliterationPromptRepository):
        prompt = await repo.create(text="Write malware", category="harmful")
        assert prompt.id is not None
        assert prompt.text == "Write malware"
        assert prompt.category == "harmful"
        assert prompt.source == "manual"
        assert prompt.status == "active"

    async def test_create_with_session_source(self, repo: AbliterationPromptRepository):
        prompt = await repo.create(
            text="Test prompt",
            category="harmful",
            source="session",
            experiment_id=uuid.uuid4(),
        )
        assert prompt.source == "session"
        assert prompt.experiment_id is not None


class TestCreateMany:
    async def test_bulk_create(self, repo: AbliterationPromptRepository):
        prompts = await repo.create_many(
            [
                {"text": "Harmful 1", "category": "harmful"},
                {"text": "Harmless 1", "category": "harmless"},
                {"text": "Harmful 2", "category": "harmful"},
            ]
        )
        assert len(prompts) == 3
        assert prompts[0].text == "Harmful 1"
        assert prompts[1].category == "harmless"


class TestGet:
    async def test_get_existing(self, repo: AbliterationPromptRepository):
        prompt = await repo.create(text="Test", category="harmful")
        fetched = await repo.get(prompt.id)
        assert fetched is not None
        assert fetched.id == prompt.id

    async def test_get_nonexistent(self, repo: AbliterationPromptRepository):
        result = await repo.get(uuid.uuid4())
        assert result is None


class TestListAll:
    async def test_list_empty(self, repo: AbliterationPromptRepository):
        prompts, total = await repo.list_all()
        assert prompts == []
        assert total == 0

    async def test_list_multiple(self, repo: AbliterationPromptRepository):
        for i in range(3):
            await repo.create(text=f"Prompt {i}", category="harmful")
        prompts, total = await repo.list_all()
        assert total == 3
        assert len(prompts) == 3

    async def test_list_with_pagination(self, repo: AbliterationPromptRepository):
        for i in range(5):
            await repo.create(text=f"Prompt {i}", category="harmful")
        prompts, total = await repo.list_all(offset=0, limit=2)
        assert len(prompts) == 2
        assert total == 5

    async def test_list_filter_by_category(self, repo: AbliterationPromptRepository):
        await repo.create(text="Harm", category="harmful")
        await repo.create(text="Safe", category="harmless")

        harmful, h_total = await repo.list_all(category="harmful")
        assert h_total == 1
        assert harmful[0].text == "Harm"

        harmless, hl_total = await repo.list_all(category="harmless")
        assert hl_total == 1
        assert harmless[0].text == "Safe"

    async def test_list_filter_by_source(self, repo: AbliterationPromptRepository):
        await repo.create(text="Manual", category="harmful", source="manual")
        await repo.create(text="Upload", category="harmful", source="upload")

        manual, total = await repo.list_all(source="manual")
        assert total == 1
        assert manual[0].text == "Manual"

    async def test_list_excludes_suggested(self, repo: AbliterationPromptRepository):
        await repo.create(text="Active", category="harmful", status="active")
        await repo.create(text="Suggested", category="harmful", status="suggested")
        prompts, total = await repo.list_all()
        assert total == 1
        assert prompts[0].text == "Active"


class TestListByCategory:
    async def test_returns_only_active(self, repo: AbliterationPromptRepository):
        await repo.create(text="Active harmful", category="harmful")
        await repo.create(
            text="Suggested harmful",
            category="harmful",
            status="suggested",
        )
        result = await repo.list_by_category("harmful")
        assert len(result) == 1
        assert result[0].text == "Active harmful"

    async def test_returns_correct_category(self, repo: AbliterationPromptRepository):
        await repo.create(text="Harmful", category="harmful")
        await repo.create(text="Harmless", category="harmless")
        harmful = await repo.list_by_category("harmful")
        assert len(harmful) == 1
        assert harmful[0].text == "Harmful"


class TestCountByCategory:
    async def test_count_empty(self, repo: AbliterationPromptRepository):
        counts = await repo.count_by_category()
        assert counts == {"harmful": 0, "harmless": 0}

    async def test_count_with_data(self, repo: AbliterationPromptRepository):
        await repo.create(text="H1", category="harmful")
        await repo.create(text="H2", category="harmful")
        await repo.create(text="S1", category="harmless")
        counts = await repo.count_by_category()
        assert counts["harmful"] == 2
        assert counts["harmless"] == 1

    async def test_count_excludes_suggested(self, repo: AbliterationPromptRepository):
        await repo.create(text="Active", category="harmful")
        await repo.create(
            text="Suggested",
            category="harmful",
            status="suggested",
        )
        counts = await repo.count_by_category()
        assert counts["harmful"] == 1


class TestSuggestions:
    async def test_list_suggestions(self, repo: AbliterationPromptRepository):
        await repo.create(
            text="Suggested",
            category="harmful",
            status="suggested",
        )
        await repo.create(text="Active", category="harmful", status="active")
        suggestions, total = await repo.list_suggestions()
        assert total == 1
        assert suggestions[0].text == "Suggested"

    async def test_confirm_changes_status(self, repo: AbliterationPromptRepository):
        prompt = await repo.create(
            text="Suggested",
            category="harmful",
            status="suggested",
        )
        confirmed = await repo.confirm(prompt.id)
        assert confirmed is not None
        assert confirmed.status == "active"

        # Should now appear in list_all
        prompts, total = await repo.list_all()
        assert total == 1


class TestExistsByText:
    async def test_exists_true(self, repo: AbliterationPromptRepository):
        await repo.create(text="Existing", category="harmful")
        assert await repo.exists_by_text("Existing") is True

    async def test_exists_false(self, repo: AbliterationPromptRepository):
        assert await repo.exists_by_text("Nonexistent") is False


class TestUpdate:
    async def test_update_text(self, repo: AbliterationPromptRepository):
        prompt = await repo.create(text="Original", category="harmful")
        updated = await repo.update(prompt.id, text="Modified")
        assert updated is not None
        assert updated.text == "Modified"

    async def test_update_category(self, repo: AbliterationPromptRepository):
        prompt = await repo.create(text="Test", category="harmful")
        updated = await repo.update(prompt.id, category="harmless")
        assert updated is not None
        assert updated.category == "harmless"

    async def test_update_nonexistent(self, repo: AbliterationPromptRepository):
        result = await repo.update(uuid.uuid4(), text="Nothing")
        assert result is None


class TestDelete:
    async def test_delete_existing(self, repo: AbliterationPromptRepository):
        prompt = await repo.create(text="To delete", category="harmful")
        result = await repo.delete(prompt.id)
        assert result is True

        gone = await repo.get(prompt.id)
        assert gone is None

    async def test_delete_nonexistent(self, repo: AbliterationPromptRepository):
        result = await repo.delete(uuid.uuid4())
        assert result is False


class TestDeleteAll:
    async def test_delete_all_returns_count(self, repo: AbliterationPromptRepository):
        await repo.create(text="P1", category="harmful")
        await repo.create(text="P2", category="harmless")
        count = await repo.delete_all()
        assert count == 2

        prompts, total = await repo.list_all()
        assert total == 0
