"""Pipeline for pulling pre-built abliterated models from Ollama Hub."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

import httpx
import structlog

logger = structlog.get_logger(__name__)

ProgressCallback = Callable[[float, str], Awaitable[None]]


async def run_pull(
    source_tag: str,
    output_name: str,
    ollama_base_url: str,
    on_progress: ProgressCallback,
) -> None:
    """Pull a model from Ollama Hub with streaming progress.

    If output_name differs from source_tag, the model is copied to
    the new name after pulling.
    """
    await on_progress(0.0, f"Pulling {source_tag}")

    async with (
        httpx.AsyncClient(base_url=ollama_base_url, timeout=1800.0) as client,
        client.stream(
            "POST",
            "/api/pull",
            json={"name": source_tag, "stream": True},
        ) as resp,
    ):
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            status = data.get("status", "")
            completed = data.get("completed", 0)
            total = data.get("total", 0)

            if total > 0:
                pct = min((completed / total) * 85.0, 85.0)
                await on_progress(pct, status)
            else:
                await on_progress(0.0, status)

    await on_progress(85.0, "Pull complete")

    # Rename if output differs from source
    if output_name and output_name != source_tag:
        from adversarial_framework.services.finetuning.ollama_import import (
            rename_ollama_model,
        )

        await rename_ollama_model(source_tag, output_name, ollama_base_url, on_progress)
    else:
        await on_progress(100.0, "Model ready")

    logger.info(
        "pull_complete",
        source=source_tag,
        output=output_name,
    )
