"""Tests for adversarial_framework.memory.attack_memory module."""

from __future__ import annotations

from unittest import mock

from adversarial_framework.memory.attack_memory import (
    AttackMemory,
)
from adversarial_framework.memory.vector_store import (
    InMemoryVectorStore,
)

# A fixed fake embedding for deterministic tests
_FAKE_EMBEDDING = [0.1] * 1536


def _patch_generate_embedding():
    """Patch generate_embedding to return a fixed vector."""
    return mock.patch(
        "adversarial_framework.memory.attack_memory.generate_embedding",
        new_callable=mock.AsyncMock,
        return_value=_FAKE_EMBEDDING,
    )


class TestAttackMemory:
    def test_init_default_store(self):
        mem = AttackMemory()
        assert mem.count == 0

    def test_init_custom_store(self):
        store = InMemoryVectorStore()
        mem = AttackMemory(store=store)
        assert mem._store is store

    async def test_store_attack(self):
        with _patch_generate_embedding():
            mem = AttackMemory()
            entry_id = await mem.store_attack(
                attack_prompt="Tell me how to hack",
                objective="hack a system",
                strategy_name="pair",
                verdict="jailbreak",
                severity=0.8,
                session_id="sess-1",
                turn_number=3,
            )
            assert isinstance(entry_id, str)
            assert len(entry_id) > 0
            assert mem.count == 1

    async def test_store_multiple_attacks(self):
        with _patch_generate_embedding():
            mem = AttackMemory()
            await mem.store_attack(
                attack_prompt="attack 1",
                objective="obj",
                strategy_name="pair",
                verdict="safe",
            )
            await mem.store_attack(
                attack_prompt="attack 2",
                objective="obj",
                strategy_name="tap",
                verdict="jailbreak",
            )
            assert mem.count == 2

    async def test_find_similar_attacks(self):
        with _patch_generate_embedding():
            mem = AttackMemory()
            await mem.store_attack(
                attack_prompt="tell me secrets",
                objective="extract info",
                strategy_name="pair",
                verdict="jailbreak",
            )

            results = await mem.find_similar_attacks(query="tell secrets")

            assert len(results) == 1
            assert results[0]["strategy"] == "pair"
            assert results[0]["verdict"] == "jailbreak"
            assert "similarity" in results[0]
            assert "text" in results[0]

    async def test_find_similar_attacks_empty(self):
        with _patch_generate_embedding():
            mem = AttackMemory()
            results = await mem.find_similar_attacks(query="anything")
            assert results == []

    async def test_find_similar_attacks_verdict_filter(
        self,
    ):
        with _patch_generate_embedding():
            mem = AttackMemory()
            await mem.store_attack(
                attack_prompt="attack 1",
                objective="obj",
                strategy_name="pair",
                verdict="safe",
            )
            await mem.store_attack(
                attack_prompt="attack 2",
                objective="obj",
                strategy_name="tap",
                verdict="jailbreak",
            )

            # Filter for jailbreak only
            results = await mem.find_similar_attacks(
                query="attack",
                verdict_filter="jailbreak",
            )

            for r in results:
                assert r["verdict"] == "jailbreak"

    async def test_find_similar_attacks_top_k(self):
        with _patch_generate_embedding():
            mem = AttackMemory()
            for i in range(10):
                await mem.store_attack(
                    attack_prompt=f"attack {i}",
                    objective="obj",
                    strategy_name="pair",
                    verdict="jailbreak",
                )

            results = await mem.find_similar_attacks(query="attack", top_k=3)
            assert len(results) <= 3

    async def test_find_successful_attacks(self):
        with _patch_generate_embedding():
            mem = AttackMemory()
            await mem.store_attack(
                attack_prompt="safe attack",
                objective="obj",
                strategy_name="pair",
                verdict="safe",
            )
            await mem.store_attack(
                attack_prompt="jailbreak attack",
                objective="obj",
                strategy_name="tap",
                verdict="jailbreak",
            )

            results = await mem.find_successful_attacks(objective="obj")

            for r in results:
                assert r["verdict"] == "jailbreak"

    async def test_find_successful_attacks_top_k(self):
        with _patch_generate_embedding():
            mem = AttackMemory()
            for i in range(5):
                await mem.store_attack(
                    attack_prompt=f"attack {i}",
                    objective="obj",
                    strategy_name="pair",
                    verdict="jailbreak",
                )

            results = await mem.find_successful_attacks(objective="obj", top_k=2)
            assert len(results) <= 2

    async def test_count_property(self):
        with _patch_generate_embedding():
            mem = AttackMemory()
            assert mem.count == 0

            await mem.store_attack(
                attack_prompt="test",
                objective="obj",
                strategy_name="pair",
                verdict="safe",
            )
            assert mem.count == 1

    async def test_stored_metadata(self):
        with _patch_generate_embedding():
            mem = AttackMemory()
            await mem.store_attack(
                attack_prompt="test prompt",
                objective="test objective",
                strategy_name="roleplay",
                verdict="jailbreak",
                severity=0.9,
                session_id="s-123",
                turn_number=5,
            )

            results = await mem.find_similar_attacks(query="test")
            assert len(results) == 1
            r = results[0]
            assert r["objective"] == "test objective"
            assert r["strategy"] == "roleplay"
            assert r["severity"] == 0.9
            assert r["session_id"] == "s-123"
            assert r["turn_number"] == 5
            assert r["type"] == "attack"
