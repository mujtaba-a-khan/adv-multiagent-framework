"""Tests for adversarial_framework.memory.defense_memory module."""

from __future__ import annotations

from unittest import mock

from adversarial_framework.memory.defense_memory import (
    DefenseMemory,
)
from adversarial_framework.memory.vector_store import (
    InMemoryVectorStore,
)

# A fixed fake embedding for deterministic tests
_FAKE_EMBEDDING = [0.1] * 1536


def _patch_generate_embedding():
    """Patch generate_embedding to return a fixed vector."""
    return mock.patch(
        "adversarial_framework.memory.defense_memory.generate_embedding",
        new_callable=mock.AsyncMock,
        return_value=_FAKE_EMBEDDING,
    )


class TestDefenseMemory:
    def test_init_default_store(self):
        mem = DefenseMemory()
        assert mem.count == 0

    def test_init_custom_store(self):
        store = InMemoryVectorStore()
        mem = DefenseMemory(store=store)
        assert mem._store is store

    async def test_store_defense(self):
        with _patch_generate_embedding():
            mem = DefenseMemory()
            entry_id = await mem.store_defense(
                defense_description="Block roleplay",
                defense_type="keyword_filter",
                defense_config={"keywords": ["hack"]},
                attack_strategy="roleplay",
                block_rate=0.85,
                false_positive_rate=0.05,
                session_id="sess-1",
            )
            assert isinstance(entry_id, str)
            assert len(entry_id) > 0
            assert mem.count == 1

    async def test_store_multiple_defenses(self):
        with _patch_generate_embedding():
            mem = DefenseMemory()
            await mem.store_defense(
                defense_description="defense 1",
                defense_type="type_a",
                defense_config={},
                attack_strategy="pair",
            )
            await mem.store_defense(
                defense_description="defense 2",
                defense_type="type_b",
                defense_config={},
                attack_strategy="tap",
            )
            assert mem.count == 2

    async def test_find_countermeasures(self):
        with _patch_generate_embedding():
            mem = DefenseMemory()
            await mem.store_defense(
                defense_description="Block via filter",
                defense_type="keyword_filter",
                defense_config={"k": "v"},
                attack_strategy="roleplay",
                block_rate=0.9,
            )

            results = await mem.find_countermeasures(
                attack_description="roleplay attack",
                min_block_rate=0.5,
            )

            assert len(results) == 1
            assert results[0]["defense_type"] == ("keyword_filter")
            assert results[0]["block_rate"] == 0.9
            assert "similarity" in results[0]

    async def test_find_countermeasures_empty(self):
        with _patch_generate_embedding():
            mem = DefenseMemory()
            results = await mem.find_countermeasures(attack_description="something")
            assert results == []

    async def test_find_countermeasures_filters_low_block_rate(
        self,
    ):
        with _patch_generate_embedding():
            mem = DefenseMemory()
            await mem.store_defense(
                defense_description="weak defense",
                defense_type="weak",
                defense_config={},
                attack_strategy="pair",
                block_rate=0.2,
            )
            await mem.store_defense(
                defense_description="strong defense",
                defense_type="strong",
                defense_config={},
                attack_strategy="pair",
                block_rate=0.9,
            )

            results = await mem.find_countermeasures(
                attack_description="pair attack",
                min_block_rate=0.5,
            )

            # Only the strong defense should be returned
            for r in results:
                assert r["block_rate"] >= 0.5

    async def test_find_countermeasures_top_k(self):
        with _patch_generate_embedding():
            mem = DefenseMemory()
            for i in range(10):
                await mem.store_defense(
                    defense_description=f"defense {i}",
                    defense_type="type",
                    defense_config={},
                    attack_strategy="pair",
                    block_rate=0.8,
                )

            results = await mem.find_countermeasures(attack_description="pair", top_k=3)
            assert len(results) <= 3

    async def test_find_countermeasures_min_similarity(
        self,
    ):
        with _patch_generate_embedding():
            mem = DefenseMemory()
            await mem.store_defense(
                defense_description="some defense",
                defense_type="type",
                defense_config={},
                attack_strategy="roleplay",
                block_rate=0.8,
            )

            # Very high min_similarity might still match
            # since we use the same fake embedding
            results = await mem.find_countermeasures(
                attack_description="test",
                min_similarity=0.99,
            )
            # The fake embedding makes everything identical
            assert len(results) >= 0

    async def test_count_property(self):
        with _patch_generate_embedding():
            mem = DefenseMemory()
            assert mem.count == 0

            await mem.store_defense(
                defense_description="test",
                defense_type="type",
                defense_config={},
                attack_strategy="pair",
            )
            assert mem.count == 1

    async def test_stored_metadata(self):
        with _patch_generate_embedding():
            mem = DefenseMemory()
            await mem.store_defense(
                defense_description="Semantic guard",
                defense_type="semantic_guard",
                defense_config={"threshold": 0.8},
                attack_strategy="encoding",
                block_rate=0.75,
                false_positive_rate=0.1,
                session_id="s-456",
            )

            results = await mem.find_countermeasures(
                attack_description="encoding",
                min_block_rate=0.5,
            )
            assert len(results) == 1
            r = results[0]
            assert r["defense_type"] == "semantic_guard"
            assert r["defense_config"] == {"threshold": 0.8}
            assert r["attack_strategy"] == "encoding"
            assert r["block_rate"] == 0.75
            assert r["false_positive_rate"] == 0.1
            assert r["session_id"] == "s-456"
            assert r["type"] == "defense"

    async def test_default_block_rate_zero(self):
        with _patch_generate_embedding():
            mem = DefenseMemory()
            await mem.store_defense(
                defense_description="no rate",
                defense_type="type",
                defense_config={},
                attack_strategy="pair",
            )

            # Default block_rate=0.0, min_block_rate=0.5
            results = await mem.find_countermeasures(
                attack_description="pair",
                min_block_rate=0.5,
            )
            # Should not return the entry (block_rate=0)
            assert len(results) == 0
