"""Import local models (safetensors) into Ollama via GGUF conversion.

Converts HuggingFace safetensors → GGUF using llama.cpp's converter,
then imports the GGUF into Ollama.  This preserves the original
tokenizer and chat template — unlike Ollama's built-in safetensors
converter which breaks non-Llama tokenizers (e.g. Mistral Tekken).

Disk-space-aware: source files are deleted after GGUF conversion
and the GGUF is deleted after Ollama import.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx
import structlog

logger = structlog.get_logger(__name__)

ProgressCallback = Callable[[float, str], Awaitable[None]]

_OLLAMA_DIR = Path.home() / ".ollama" / "models"

# ── GGUF converter management ────────────────────────────────────

_CONVERTER_DIR = Path.home() / ".cache" / "adv-framework"
_CONVERTER_SCRIPT = _CONVERTER_DIR / "convert_hf_to_gguf.py"
# Pinned to commit 381174bb (gguf==0.17.1 release, 2025-06-19).
# Must match the installed gguf PyPI package version — master
# may use imports that don't exist in the published package yet.
_CONVERTER_URL = (
    "https://raw.githubusercontent.com/ggml-org/llama.cpp/"
    "381174bbdaf1/convert_hf_to_gguf.py"
)


async def _ensure_gguf_converter(
    *, force: bool = False,
) -> Path:
    """Download convert_hf_to_gguf.py from llama.cpp if not cached.

    The script handles all HF architectures natively (Llama, Mistral,
    Qwen, Gemma, Phi, etc.) and correctly converts tokenizers.
    """
    if _CONVERTER_SCRIPT.exists() and not force:
        return _CONVERTER_SCRIPT

    _CONVERTER_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("downloading_gguf_converter", url=_CONVERTER_URL)

    async with httpx.AsyncClient(
        timeout=60.0, follow_redirects=True,
    ) as client:
        resp = await client.get(_CONVERTER_URL)
        resp.raise_for_status()

    await asyncio.to_thread(
        _CONVERTER_SCRIPT.write_text, resp.text,
    )
    logger.info(
        "gguf_converter_cached", path=str(_CONVERTER_SCRIPT),
    )
    return _CONVERTER_SCRIPT


# ── GGUF conversion ─────────────────────────────────────────────


async def _run_converter(
    converter: Path,
    model_dir: Path,
    gguf_path: Path,
) -> tuple[int, str, str]:
    """Run the converter subprocess, return (code, stdout, stderr)."""
    cmd = [
        sys.executable,
        str(converter),
        str(model_dir),
        "--outfile",
        str(gguf_path),
        "--outtype",
        "f16",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    raw_out, raw_err = await proc.communicate()
    return (
        proc.returncode or 0,
        raw_out.decode(errors="replace"),
        raw_err.decode(errors="replace"),
    )


async def _convert_hf_to_gguf(
    model_dir: Path,
    gguf_path: Path,
    on_progress: ProgressCallback,
) -> None:
    """Convert HuggingFace safetensors to F16 GGUF.

    Uses llama.cpp's convert_hf_to_gguf.py which handles all
    architectures natively and preserves the correct tokenizer
    and chat template in the GGUF metadata.

    If the cached converter script fails with an import error,
    it is re-downloaded and retried once (handles stale cache
    after a ``gguf`` package upgrade).
    """
    converter = await _ensure_gguf_converter()

    await on_progress(80.0, "Converting to GGUF (F16)")
    logger.info(
        "gguf_conversion_start",
        model_dir=str(model_dir),
        output=str(gguf_path),
    )

    code, out, err = await _run_converter(
        converter, model_dir, gguf_path,
    )

    # Retry once if the cached script is stale (import errors)
    if code != 0 and (
        "ImportError" in err or "ModuleNotFoundError" in err
    ):
        logger.warning(
            "converter_stale_redownloading", error=err[:200],
        )
        converter = await _ensure_gguf_converter(force=True)
        code, out, err = await _run_converter(
            converter, model_dir, gguf_path,
        )

    if code != 0:
        # Truncate very long error messages
        msg = err.strip()
        if len(msg) > 500:
            msg = msg[-500:]
        raise RuntimeError(
            f"GGUF conversion failed (exit {code}): {msg}"
        )

    gguf_gb = gguf_path.stat().st_size / (1024**3)
    logger.info(
        "gguf_conversion_complete",
        gguf_size_gb=f"{gguf_gb:.1f}",
    )
    await on_progress(
        88.0, f"GGUF created ({gguf_gb:.1f} GB)",
    )


# ── Ollama model creation ───────────────────────────────────────


async def _create_ollama_model(
    gguf_path: Path,
    model_name: str,
    on_progress: ProgressCallback,
    *,
    quantize: str | None = None,
) -> None:
    """Create an Ollama model from a GGUF file via CLI.

    Writes a Modelfile pointing to the GGUF and runs
    ``ollama create``.  Optionally quantizes (e.g. q4_K_M).
    """
    modelfile = gguf_path.parent / "Modelfile"
    await asyncio.to_thread(
        modelfile.write_text, f"FROM {gguf_path}\n",
    )

    q_label = f" ({quantize})" if quantize else ""
    await on_progress(
        90.0, f"Creating {model_name}{q_label}",
    )

    cmd = [
        "ollama", "create", model_name,
        "-f", str(modelfile),
    ]
    if quantize:
        cmd.extend(["--quantize", quantize])

    logger.info(
        "ollama_create_start",
        model_name=model_name,
        quantize=quantize,
    )

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    raw_out, raw_err = await proc.communicate()
    code = proc.returncode or 0

    if code != 0:
        err = raw_err.decode(errors="replace").strip()
        if not err:
            err = raw_out.decode(errors="replace").strip()
        raise RuntimeError(
            f"ollama create failed (exit {code}): {err}"
        )

    logger.info(
        "ollama_create_complete",
        model_name=model_name,
        output=raw_out.decode(errors="replace").strip()[:200],
    )
    await on_progress(98.0, f"Model {model_name} created")


# ── Main entry point ────────────────────────────────────────────


async def import_to_ollama(
    model_path: str,
    model_name: str,
    ollama_base_url: str,
    on_progress: ProgressCallback,
    *,
    quantize: str | None = None,
) -> None:
    """Import a HuggingFace safetensors directory into Ollama.

    Pipeline:
    1. Convert safetensors → F16 GGUF (llama.cpp converter)
    2. Delete source safetensors (free disk)
    3. Create Ollama model from GGUF (optional quantization)
    4. Delete GGUF file (free disk)
    5. Verify model via API
    """
    model_dir = Path(model_path)
    gguf_name = model_name.replace(":", "-")
    gguf_path = model_dir.parent / f"{gguf_name}.gguf"

    try:
        # 1. Convert to GGUF
        await _convert_hf_to_gguf(
            model_dir, gguf_path, on_progress,
        )

        # 2. Free source safetensors
        src_bytes = sum(
            f.stat().st_size
            for f in model_dir.iterdir()
            if f.is_file()
        )
        await asyncio.to_thread(
            shutil.rmtree, str(model_dir), True,
        )
        logger.info(
            "source_files_freed",
            freed_gb=f"{src_bytes / (1024**3):.1f}",
        )

        # 3. Create Ollama model
        await _create_ollama_model(
            gguf_path, model_name, on_progress,
            quantize=quantize,
        )

    finally:
        # 4. Free GGUF file
        if gguf_path.exists():
            gb = gguf_path.stat().st_size / (1024**3)
            await asyncio.to_thread(
                gguf_path.unlink, missing_ok=True,
            )
            logger.info(
                "gguf_file_freed", freed_gb=f"{gb:.1f}",
            )

    # 5. Verify model
    async with httpx.AsyncClient(
        base_url=ollama_base_url, timeout=30.0,
    ) as client:
        resp = await client.post(
            "/api/show", json={"name": model_name},
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Model '{model_name}' not found in Ollama "
                "after create reported success"
            )

    await on_progress(100.0, "Model imported into Ollama")
    logger.info(
        "ollama_import_complete",
        model_name=model_name,
        model_path=model_path,
    )


# ── Disk management (user-triggered via API) ─────────────────────


def get_disk_status() -> dict:
    """Return disk usage summary for the Workshop UI.

    Checks both the system disk and Ollama's blob storage for orphans.
    """
    blobs_dir = _OLLAMA_DIR / "blobs"
    manifests_dir = _OLLAMA_DIR / "manifests"

    # System disk free
    try:
        usage = shutil.disk_usage("/")
        disk_total_gb = round(usage.total / (1024**3), 1)
        disk_free_gb = round(usage.free / (1024**3), 1)
    except OSError:
        disk_total_gb = 0.0
        disk_free_gb = 0.0

    # Ollama storage total
    ollama_total_bytes = 0
    if blobs_dir.exists():
        for name in os.listdir(blobs_dir):
            try:
                ollama_total_bytes += (
                    (blobs_dir / name).stat().st_size
                )
            except OSError:
                continue
    ollama_total_gb = round(ollama_total_bytes / (1024**3), 1)

    # Count orphan blobs
    referenced = _collect_referenced_digests(manifests_dir)
    orphan_count = 0
    orphan_bytes = 0
    if blobs_dir.exists():
        for name in os.listdir(blobs_dir):
            digest = name.replace("-", ":")
            if digest not in referenced:
                try:
                    orphan_bytes += (
                        (blobs_dir / name).stat().st_size
                    )
                    orphan_count += 1
                except OSError:
                    continue

    return {
        "disk_total_gb": disk_total_gb,
        "disk_free_gb": disk_free_gb,
        "ollama_storage_gb": ollama_total_gb,
        "orphan_count": orphan_count,
        "orphan_gb": round(orphan_bytes / (1024**3), 1),
    }


def cleanup_orphan_blobs() -> dict:
    """Delete Ollama blobs not referenced by any model manifest.

    Returns a summary with orphan count and freed bytes.
    """
    blobs_dir = _OLLAMA_DIR / "blobs"
    manifests_dir = _OLLAMA_DIR / "manifests"

    if not blobs_dir.exists():
        return {
            "orphan_count": 0,
            "freed_bytes": 0,
            "freed_gb": 0.0,
        }

    referenced = _collect_referenced_digests(manifests_dir)

    orphan_count = 0
    freed_bytes = 0
    for name in os.listdir(blobs_dir):
        digest = name.replace("-", ":")
        if digest not in referenced:
            blob_path = blobs_dir / name
            try:
                size = blob_path.stat().st_size
                blob_path.unlink()
                orphan_count += 1
                freed_bytes += size
                logger.info(
                    "deleted_orphan_blob",
                    digest=digest[:30],
                    size_mb=f"{size / (1024**2):.0f}",
                )
            except OSError:
                continue

    return {
        "orphan_count": orphan_count,
        "freed_bytes": freed_bytes,
        "freed_gb": round(freed_bytes / (1024**3), 1),
    }


def _collect_referenced_digests(
    manifests_dir: Path,
) -> set[str]:
    """Walk Ollama manifests and collect referenced blob digests."""
    referenced: set[str] = set()
    if not manifests_dir.exists():
        return referenced
    for root, _dirs, files in os.walk(manifests_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                with open(fpath) as fh:
                    manifest = json.load(fh)
                cfg_d = manifest.get(
                    "config", {},
                ).get("digest", "")
                if cfg_d:
                    referenced.add(cfg_d)
                for layer in manifest.get("layers", []):
                    d = layer.get("digest", "")
                    if d:
                        referenced.add(d)
            except Exception:  # noqa: BLE001
                continue
    return referenced


async def rename_ollama_model(
    source_tag: str,
    target_name: str,
    ollama_base_url: str,
    on_progress: ProgressCallback,
) -> None:
    """Copy an Ollama model to a new name."""
    await on_progress(90.0, f"Renaming model to {target_name}")

    async with httpx.AsyncClient(
        base_url=ollama_base_url, timeout=300.0,
    ) as client:
        resp = await client.post(
            "/api/create",
            json={
                "model": target_name,
                "from": source_tag,
            },
        )
        resp.raise_for_status()

    await on_progress(100.0, "Model renamed")
