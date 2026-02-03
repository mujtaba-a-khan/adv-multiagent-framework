"""Defense memory it stores and retrieves past defense patterns.

The defense memory allows the Defender agent to query for known
countermeasures against specific attack techniques.
"""

from __future__ import annotations

import uuid
from typing import Any

import structlog

from adversarial_framework.memory.embeddings import generate_embedding
from adversarial_framework.memory.vector_store import InMemoryVectorStore, VectorEntry

logger = structlog.get_logger(__name__)


class DefenseMemory:
    """Vector-backed memory for defense patterns and their effectiveness."""

    def __init__(self, store: InMemoryVectorStore | None = None) -> None:
        self._store = store or InMemoryVectorStore()

    async def store_defense(
        self,
        defense_description: str,
        defense_type: str,
        defense_config: dict[str, Any],
        attack_strategy: str,
        block_rate: float = 0.0,
        false_positive_rate: float = 0.0,
        session_id: str = "",
    ) -> str:
        """Store a defense pattern with its effectiveness metrics.

        Returns:
            The ID of the stored entry.
        """
        entry_id = str(uuid.uuid4())
        embedding = await generate_embedding(defense_description)

        entry = VectorEntry(
            id=entry_id,
            embedding=embedding,
            text=defense_description,
            metadata={
                "defense_type": defense_type,
                "defense_config": defense_config,
                "attack_strategy": attack_strategy,
                "block_rate": block_rate,
                "false_positive_rate": false_positive_rate,
                "session_id": session_id,
                "type": "defense",
            },
        )

        self._store.add(entry)
        logger.info(
            "defense_stored",
            entry_id=entry_id,
            defense_type=defense_type,
            block_rate=block_rate,
        )
        return entry_id

    async def find_countermeasures(
        self,
        attack_description: str,
        top_k: int = 3,
        min_similarity: float = 0.3,
        min_block_rate: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Find effective defenses against similar attacks.

        Args:
            attack_description: Description of the attack to defend against.
            top_k: Number of results.
            min_similarity: Minimum cosine similarity.
            min_block_rate: Minimum block rate for returned defenses.

        Returns:
            List of dicts with defense details and similarity scores.
        """
        query_embedding = await generate_embedding(attack_description)

        results = self._store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Over-fetch to filter by block rate
            min_similarity=min_similarity,
            filter_metadata={"type": "defense"},
        )

        filtered = [
            {
                "id": entry.id,
                "text": entry.text,
                "similarity": score,
                **entry.metadata,
            }
            for entry, score in results
            if entry.metadata.get("block_rate", 0) >= min_block_rate
        ]

        return filtered[:top_k]

    @property
    def count(self) -> int:
        """Number of stored defenses."""
        return self._store.count()
