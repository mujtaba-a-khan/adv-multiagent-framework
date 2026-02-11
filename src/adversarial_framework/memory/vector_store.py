"""In-memory vector store with pgvector integration support.

Provides an in-memory vector store for development and a pgvector-backed
store for production.  Both implement the same interface for similarity search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from adversarial_framework.memory.embeddings import cosine_similarity

logger = structlog.get_logger(__name__)


@dataclass
class VectorEntry:
    """A single entry in the vector store."""

    id: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    text: str = ""


class InMemoryVectorStore:
    """Simple in-memory vector store for development and testing.

    For production, use `PgVectorStore` which stores embeddings in
    PostgreSQL with pgvector HNSW indexes.
    """

    def __init__(self) -> None:
        self._entries: dict[str, VectorEntry] = {}

    def add(self, entry: VectorEntry) -> None:
        """Add or update an entry."""
        self._entries[entry.id] = entry
        logger.debug("vector_store_add", id=entry.id)

    def add_batch(self, entries: list[VectorEntry]) -> None:
        """Add multiple entries."""
        for entry in entries:
            self._entries[entry.id] = entry
        logger.debug("vector_store_batch_add", count=len(entries))

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        min_similarity: float = 0.0,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[tuple[VectorEntry, float]]:
        """Find the top-k most similar entries.

        Args:
            query_embedding: The query vector.
            top_k: Number of results to return.
            min_similarity: Minimum cosine similarity threshold.
            filter_metadata: Optional metadata filter (exact match).

        Returns:
            List of (entry, similarity_score) tuples, sorted by similarity desc.
        """
        scored: list[tuple[VectorEntry, float]] = []

        for entry in self._entries.values():
            if filter_metadata and not all(
                entry.metadata.get(k) == v for k, v in filter_metadata.items()
            ):
                continue

            sim = cosine_similarity(query_embedding, entry.embedding)
            if sim >= min_similarity:
                scored.append((entry, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def delete(self, entry_id: str) -> bool:
        """Delete an entry by ID."""
        if entry_id in self._entries:
            del self._entries[entry_id]
            return True
        return False

    def count(self) -> int:
        """Return the number of entries."""
        return len(self._entries)

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()
