"""Pipeline for Supervised Fine-Tuning (SFT) with QLoRA via mlx-lm.

This pipeline runs mlx_lm.lora as a subprocess and parses progress from stdout.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path

import structlog

from adversarial_framework.services.finetuning.memory import (
    check_memory_feasibility,
)

logger = structlog.get_logger(__name__)

ProgressCallback = Callable[[float, str], Awaitable[None]]


async def run_sft(
    source_model: str,
    output_name: str,
    config: dict,
    ollama_base_url: str,
    on_progress: ProgressCallback,
) -> None:
    """Run the SFT QLoRA pipeline.

    Steps:
    1. Memory check (reject 14B+ on 16GB)
    2. Stop running Ollama models
    3. Run mlx_lm.lora training
    4. Fuse adapter
    5. Convert to GGUF
    6. Import to Ollama
    """
    # Step 1: Memory check
    await on_progress(0.0, "Checking memory feasibility")
    feasible, reason = check_memory_feasibility(
        source_model, "sft"
    )
    if not feasible:
        raise RuntimeError(f"Insufficient memory: {reason}")

    # Step 2: Stop running Ollama models
    await on_progress(2.0, "Stopping running Ollama models")
    from adversarial_framework.services.finetuning.abliterate_pipeline import (
        _stop_ollama_models,
    )

    await _stop_ollama_models(ollama_base_url)

    dataset_path = config.get("dataset_path")
    if not dataset_path:
        raise RuntimeError("SFT requires a dataset_path in config")

    if not Path(dataset_path).exists():
        raise RuntimeError(
            f"Dataset not found: {dataset_path}"
        )

    lora_rank = config.get("lora_rank", 8)
    lora_alpha = config.get("lora_alpha", 16)
    batch_size = config.get("batch_size", 1)
    lora_layers = config.get("lora_layers", 8)
    iters = config.get("iters", 500)

    work_dir = Path(tempfile.mkdtemp(prefix="adv-sft-"))
    adapter_path = work_dir / "adapters"
    fused_path = work_dir / "fused"

    try:
        # Step 3: Training
        await on_progress(5.0, "Starting QLoRA training")
        cmd = [
            "python",
            "-m",
            "mlx_lm.lora",
            "--model",
            source_model,
            "--data",
            dataset_path,
            "--train",
            "--adapter-path",
            str(adapter_path),
            "--batch-size",
            str(batch_size),
            "--lora-layers",
            str(lora_layers),
            "--iters",
            str(iters),
            "--lora-r",
            str(lora_rank),
            "--lora-alpha",
            str(lora_alpha),
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        if process.stdout:
            async for line_bytes in process.stdout:
                line = line_bytes.decode("utf-8", errors="replace")
                # Parse training progress from mlx-lm output
                if "iter" in line.lower() and "/" in line:
                    # Estimate progress from iteration count
                    pct = min(5.0 + 75.0, 80.0)
                    await on_progress(pct, line.strip()[:200])

        returncode = await process.wait()
        if returncode != 0:
            raise RuntimeError(
                f"mlx_lm.lora failed with code {returncode}"
            )

        await on_progress(80.0, "Training complete")

        # Step 4: Fuse adapter
        await on_progress(82.0, "Fusing LoRA adapter")
        fuse_cmd = [
            "python",
            "-m",
            "mlx_lm.fuse",
            "--model",
            source_model,
            "--adapter-path",
            str(adapter_path),
            "--save-path",
            str(fused_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *fuse_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        await proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("mlx_lm.fuse failed")

        await on_progress(86.0, "Adapter fused")

        # Step 5: Import to Ollama (handles conversion + quantization)
        from adversarial_framework.services.finetuning.ollama_import import (
            import_to_ollama,
        )

        quantize = config.get("quantize", "q4_K_M")
        await import_to_ollama(
            str(fused_path),
            output_name,
            ollama_base_url,
            on_progress,
            quantize=quantize,
        )

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    logger.info(
        "sft_complete", source=source_model, output=output_name
    )
