"""Tests for adversarial_framework.memory.vector_store module."""

from __future__ import annotations

from adversarial_framework.memory.vector_store import (
    InMemoryVectorStore,
    VectorEntry,
)


def _make_entry(
    entry_id: str,
    embedding: list[float] | None = None,
    metadata: dict | None = None,
    text: str = "",
) -> VectorEntry:
    """Helper to create a VectorEntry."""
    return VectorEntry(
        id=entry_id,
        embedding=embedding or [1.0, 0.0, 0.0],
        metadata=metadata or {},
        text=text,
    )


# VectorEntry dataclass


class TestVectorEntry:
    def test_creation(self):
        e = VectorEntry(
            id="1",
            embedding=[0.1, 0.2],
            metadata={"key": "val"},
            text="hello",
        )
        assert e.id == "1"
        assert e.embedding == [0.1, 0.2]
        assert e.metadata == {"key": "val"}
        assert e.text == "hello"

    def test_defaults(self):
        e = VectorEntry(id="2", embedding=[1.0])
        assert e.metadata == {}
        assert e.text == ""


# InMemoryVectorStore


class TestInMemoryVectorStore:
    def test_empty_store(self):
        store = InMemoryVectorStore()
        assert store.count() == 0

    def test_add_entry(self):
        store = InMemoryVectorStore()
        store.add(_make_entry("e1"))
        assert store.count() == 1

    def test_add_overwrites_same_id(self):
        store = InMemoryVectorStore()
        store.add(_make_entry("e1", text="first"))
        store.add(_make_entry("e1", text="second"))
        assert store.count() == 1

    def test_add_batch(self):
        store = InMemoryVectorStore()
        entries = [
            _make_entry("a"),
            _make_entry("b"),
            _make_entry("c"),
        ]
        store.add_batch(entries)
        assert store.count() == 3

    def test_delete_existing(self):
        store = InMemoryVectorStore()
        store.add(_make_entry("e1"))
        assert store.delete("e1") is True
        assert store.count() == 0

    def test_delete_nonexistent(self):
        store = InMemoryVectorStore()
        assert store.delete("nope") is False

    def test_clear(self):
        store = InMemoryVectorStore()
        store.add_batch([_make_entry("a"), _make_entry("b")])
        store.clear()
        assert store.count() == 0

    def test_search_empty_store(self):
        store = InMemoryVectorStore()
        results = store.search([1.0, 0.0, 0.0])
        assert results == []

    def test_search_returns_sorted_by_similarity(self):
        store = InMemoryVectorStore()
        # e1 is identical to query
        store.add(_make_entry("e1", embedding=[1.0, 0.0, 0.0]))
        # e2 is orthogonal
        store.add(_make_entry("e2", embedding=[0.0, 1.0, 0.0]))
        # e3 is partially aligned
        store.add(_make_entry("e3", embedding=[0.7, 0.7, 0.0]))

        results = store.search([1.0, 0.0, 0.0], top_k=3)
        assert len(results) == 3
        # Most similar first
        assert results[0][0].id == "e1"
        # Similarity scores are descending
        assert results[0][1] >= results[1][1]
        assert results[1][1] >= results[2][1]

    def test_search_top_k(self):
        store = InMemoryVectorStore()
        for i in range(10):
            store.add(
                _make_entry(
                    f"e{i}",
                    embedding=[float(i), 0.0, 0.0],
                )
            )

        results = store.search([1.0, 0.0, 0.0], top_k=3)
        assert len(results) <= 3

    def test_search_min_similarity(self):
        store = InMemoryVectorStore()
        store.add(_make_entry("e1", embedding=[1.0, 0.0, 0.0]))
        store.add(_make_entry("e2", embedding=[0.0, 1.0, 0.0]))

        # Only e1 should match with high threshold
        results = store.search([1.0, 0.0, 0.0], min_similarity=0.9)
        assert len(results) == 1
        assert results[0][0].id == "e1"

    def test_search_filter_metadata(self):
        store = InMemoryVectorStore()
        store.add(
            _make_entry(
                "e1",
                embedding=[1.0, 0.0, 0.0],
                metadata={"type": "attack"},
            )
        )
        store.add(
            _make_entry(
                "e2",
                embedding=[1.0, 0.0, 0.0],
                metadata={"type": "defense"},
            )
        )

        results = store.search(
            [1.0, 0.0, 0.0],
            filter_metadata={"type": "attack"},
        )
        assert len(results) == 1
        assert results[0][0].id == "e1"

    def test_search_filter_metadata_no_match(self):
        store = InMemoryVectorStore()
        store.add(
            _make_entry(
                "e1",
                embedding=[1.0, 0.0, 0.0],
                metadata={"type": "attack"},
            )
        )

        results = store.search(
            [1.0, 0.0, 0.0],
            filter_metadata={"type": "nonexistent"},
        )
        assert len(results) == 0

    def test_search_filter_metadata_multiple_keys(self):
        store = InMemoryVectorStore()
        store.add(
            _make_entry(
                "e1",
                embedding=[1.0, 0.0, 0.0],
                metadata={
                    "type": "attack",
                    "verdict": "jailbreak",
                },
            )
        )
        store.add(
            _make_entry(
                "e2",
                embedding=[1.0, 0.0, 0.0],
                metadata={
                    "type": "attack",
                    "verdict": "safe",
                },
            )
        )

        results = store.search(
            [1.0, 0.0, 0.0],
            filter_metadata={
                "type": "attack",
                "verdict": "jailbreak",
            },
        )
        assert len(results) == 1
        assert results[0][0].id == "e1"

    def test_search_no_filter(self):
        store = InMemoryVectorStore()
        store.add(
            _make_entry(
                "e1",
                embedding=[1.0, 0.0, 0.0],
                metadata={"type": "attack"},
            )
        )
        store.add(
            _make_entry(
                "e2",
                embedding=[0.9, 0.1, 0.0],
                metadata={"type": "defense"},
            )
        )

        results = store.search([1.0, 0.0, 0.0], filter_metadata=None)
        assert len(results) == 2
