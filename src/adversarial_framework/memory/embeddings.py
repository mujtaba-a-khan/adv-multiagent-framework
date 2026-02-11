"""Embedding generation service for attack prompts and defense patterns.

Uses the LLM provider (Ollama) to generate embeddings via the /api/embeddings
endpoint.  Falls back to a simple TF-IDF-like hash if the provider is unavailable.
"""

from __future__ import annotations

import hashlib
import struct

import httpx
import structlog

logger = structlog.get_logger(__name__)

_EMBEDDING_DIM = 1536  # Standard dimension for compatibility


async def generate_embedding(
    text: str,
    model: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434",
) -> list[float]:
    """Generate an embedding vector for the given text using Ollama.

    Falls back to a deterministic hash-based embedding if Ollama is unavailable.

    Args:
        text: The text to embed.
        model: Ollama embedding model name.
        base_url: Ollama base URL.

    Returns:
        A list of floats representing the embedding vector.
    """
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
            resp = await client.post(
                "/api/embeddings",
                json={"model": model, "prompt": text},
            )
            resp.raise_for_status()
            data = resp.json()
            embedding = data.get("embedding", [])
            if embedding:
                return embedding
    except (httpx.HTTPError, Exception) as exc:
        logger.warning("embedding_ollama_fallback", error=str(exc))

    # Fallback: deterministic hash-based pseudo-embedding
    return _hash_embedding(text)


def _hash_embedding(text: str, dim: int = _EMBEDDING_DIM) -> list[float]:
    """Generate a deterministic pseudo-embedding from text using SHA-256 hashing.

    This is NOT a semantic embedding only for fallback when Ollama is unavailable.
    """
    text_bytes = text.encode("utf-8")
    result: list[float] = []

    for i in range(0, dim, 8):
        chunk_hash = hashlib.sha256(text_bytes + struct.pack(">I", i)).digest()
        for j in range(min(8, dim - i)):
            # Map each byte to [-1.0, 1.0]
            result.append((chunk_hash[j] - 128) / 128.0)

    return result[:dim]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0

    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)
