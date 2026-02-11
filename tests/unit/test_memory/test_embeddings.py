"""Tests for adversarial_framework.memory.embeddings module."""

from __future__ import annotations

from unittest import mock

import pytest

from adversarial_framework.memory.embeddings import (
    _hash_embedding,
    cosine_similarity,
    generate_embedding,
)

# _hash_embedding


class TestHashEmbedding:
    def test_returns_correct_dimension(self):
        emb = _hash_embedding("test text")
        assert len(emb) == 1536

    def test_custom_dimension(self):
        emb = _hash_embedding("test", dim=128)
        assert len(emb) == 128

    def test_values_in_range(self):
        emb = _hash_embedding("hello world")
        for v in emb:
            assert -1.0 <= v <= 1.0

    def test_deterministic(self):
        e1 = _hash_embedding("same input")
        e2 = _hash_embedding("same input")
        assert e1 == e2

    def test_different_inputs_differ(self):
        e1 = _hash_embedding("input one")
        e2 = _hash_embedding("input two")
        assert e1 != e2

    def test_empty_string(self):
        emb = _hash_embedding("")
        assert len(emb) == 1536

    def test_small_dim(self):
        emb = _hash_embedding("text", dim=1)
        assert len(emb) == 1


# cosine_similarity


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_empty_vectors(self):
        assert cosine_similarity([], []) == 0.0

    def test_different_lengths(self):
        a = [1.0, 2.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0

    def test_zero_vector(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0

    def test_both_zero_vectors(self):
        a = [0.0, 0.0]
        b = [0.0, 0.0]
        assert cosine_similarity(a, b) == 0.0

    def test_normalized_vectors(self):
        a = [0.6, 0.8]
        b = [0.8, 0.6]
        sim = cosine_similarity(a, b)
        assert 0.0 < sim < 1.0

    def test_hash_self_similarity(self):
        emb = _hash_embedding("test text")
        sim = cosine_similarity(emb, emb)
        assert sim == pytest.approx(1.0, abs=1e-6)


# generate_embedding


class TestGenerateEmbedding:
    @mock.patch("adversarial_framework.memory.embeddings.httpx")
    async def test_success_from_ollama(self, mock_httpx):
        expected = [0.1] * 1536
        mock_response = mock.MagicMock()
        mock_response.json.return_value = {"embedding": expected}
        mock_response.raise_for_status = mock.MagicMock()

        mock_client = mock.AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = mock.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mock.AsyncMock(return_value=False)
        mock_httpx.AsyncClient.return_value = mock_client

        result = await generate_embedding("test text")
        assert result == expected

    @mock.patch("adversarial_framework.memory.embeddings.httpx")
    async def test_fallback_on_http_error(self, mock_httpx):
        mock_httpx.HTTPError = Exception
        mock_client = mock.AsyncMock()
        mock_client.post.side_effect = Exception("connection refused")
        mock_client.__aenter__ = mock.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mock.AsyncMock(return_value=False)
        mock_httpx.AsyncClient.return_value = mock_client

        result = await generate_embedding("test text")

        # Should fallback to hash embedding
        assert len(result) == 1536
        assert result == _hash_embedding("test text")

    @mock.patch("adversarial_framework.memory.embeddings.httpx")
    async def test_fallback_on_empty_embedding(self, mock_httpx):
        mock_response = mock.MagicMock()
        mock_response.json.return_value = {"embedding": []}
        mock_response.raise_for_status = mock.MagicMock()

        mock_client = mock.AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = mock.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mock.AsyncMock(return_value=False)
        mock_httpx.AsyncClient.return_value = mock_client

        result = await generate_embedding("test text")
        assert result == _hash_embedding("test text")

    @mock.patch("adversarial_framework.memory.embeddings.httpx")
    async def test_custom_model_and_url(self, mock_httpx):
        expected = [0.5] * 768
        mock_response = mock.MagicMock()
        mock_response.json.return_value = {"embedding": expected}
        mock_response.raise_for_status = mock.MagicMock()

        mock_client = mock.AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = mock.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mock.AsyncMock(return_value=False)
        mock_httpx.AsyncClient.return_value = mock_client

        result = await generate_embedding(
            "hello",
            model="custom-embed",
            base_url="http://gpu:11434",
        )
        assert result == expected
