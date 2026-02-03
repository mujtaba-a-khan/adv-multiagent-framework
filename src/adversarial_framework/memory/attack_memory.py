"""Attack memory stores and retrieves past attack prompts for strategy selection.

The attack memory allows the Attacker Planner to query for similar past
attacks and learn from successful techniques against similar objectives.
"""

from __future__ import annotations

import uuid
from typing import Any

import structlog

from adversarial_framework.memory.embeddings import generate_embedding
from adversarial_framework.memory.vector_store import InMemoryVectorStore, VectorEntry

logger = structlog.get_logger(__name__)


class AttackMemory:
    """Vector-backed memory for attack prompts and their outcomes."""

    def __init__(self, store: InMemoryVectorStore | None = None) -> None:
        self._store = store or InMemoryVectorStore()

    async def store_attack(
        self,
        attack_prompt: str,
        objective: str,
        strategy_name: str,
        verdict: str,
        severity: float = 0.0,
        session_id: str = "",
        turn_number: int = 0,
    ) -> str:
        """Store an attack prompt with its outcome for future retrieval.

        Returns:
            The ID of the stored entry.
        """
        entry_id = str(uuid.uuid4())
        embedding = await generate_embedding(attack_prompt)

        entry = VectorEntry(
            id=entry_id,
            embedding=embedding,
            text=attack_prompt,
            metadata={
                "objective": objective,
                "strategy": strategy_name,
                "verdict": verdict,
                "severity": severity,
                "session_id": session_id,
                "turn_number": turn_number,
                "type": "attack",
            },
        )

        self._store.add(entry)
        logger.info(
            "attack_stored",
            entry_id=entry_id,
            strategy=strategy_name,
            verdict=verdict,
        )
        return entry_id

    async def find_similar_attacks(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.3,
        verdict_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find similar past attacks.

        Args:
            query: The text to search for (attack prompt or objective).
            top_k: Number of results.
            min_similarity: Minimum cosine similarity.
            verdict_filter: Optional filter by verdict (e.g., "jailbreak").

        Returns:
            List of dicts with attack details and similarity scores.
        """
        query_embedding = await generate_embedding(query)

        filter_meta = {"type": "attack"}
        if verdict_filter:
            filter_meta["verdict"] = verdict_filter

        results = self._store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            min_similarity=min_similarity,
            filter_metadata=filter_meta,
        )

        return [
            {
                "id": entry.id,
                "text": entry.text,
                "similarity": score,
                **entry.metadata,
            }
            for entry, score in results
        ]

    async def find_successful_attacks(
        self,
        objective: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Find past successful jailbreak attacks similar to the objective."""
        return await self.find_similar_attacks(
            query=objective,
            top_k=top_k,
            verdict_filter="jailbreak",
        )

    @property
    def count(self) -> int:
        """Number of stored attacks."""
        return self._store.count()
