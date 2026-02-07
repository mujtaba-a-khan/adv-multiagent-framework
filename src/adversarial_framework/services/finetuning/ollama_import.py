"""Import GGUF models into Ollama."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import httpx
import structlog

logger = structlog.get_logger(__name__)

ProgressCallback = Callable[[float, str], Awaitable[None]]


async def import_to_ollama(
    gguf_path: str,
    model_name: str,
    ollama_base_url: str,
    on_progress: ProgressCallback,
) -> None:
    """Create a Modelfile and import the GGUF into Ollama."""
    await on_progress(90.0, "Creating Ollama Modelfile")

    modelfile_content = f"FROM {gguf_path}\n"

    await on_progress(92.0, f"Importing {model_name} into Ollama")

    async with httpx.AsyncClient(
        base_url=ollama_base_url, timeout=600.0
    ) as client:
        resp = await client.post(
            "/api/create",
            json={
                "name": model_name,
                "modelfile": modelfile_content,
            },
        )
        resp.raise_for_status()

    await on_progress(100.0, "Model imported into Ollama")
    logger.info(
        "ollama_import_complete",
        model_name=model_name,
        gguf_path=gguf_path,
    )


async def rename_ollama_model(
    source_tag: str,
    target_name: str,
    ollama_base_url: str,
    on_progress: ProgressCallback,
) -> None:
    """Copy an Ollama model to a new name via Modelfile FROM."""
    await on_progress(90.0, f"Renaming model to {target_name}")

    async with httpx.AsyncClient(
        base_url=ollama_base_url, timeout=300.0
    ) as client:
        resp = await client.post(
            "/api/create",
            json={
                "name": target_name,
                "modelfile": f"FROM {source_tag}\n",
            },
        )
        resp.raise_for_status()

    await on_progress(100.0, "Model renamed")
