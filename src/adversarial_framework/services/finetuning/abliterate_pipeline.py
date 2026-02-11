"""Pipeline for custom abliteration (refusal direction removal).

Abliteration removes the "refusal direction" from a model's residual
stream activations, following Arditi et al. (ICLR 2025).

Requires optional dependencies: transformers, torch.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path

import structlog

from adversarial_framework.services.finetuning.memory import (
    check_disk_feasibility,
    check_memory_feasibility,
)

logger = structlog.get_logger(__name__)

ProgressCallback = Callable[[float, str], Awaitable[None]]

# Regex to extract layer index from parameter names like
# "model.layers.12.self_attn.o_proj.weight"
_LAYER_NUM_RE = re.compile(r"\.layers\.(\d+)\.")


async def run_abliterate(
    source_model: str,
    output_name: str,
    config: dict,
    ollama_base_url: str,
    on_progress: ProgressCallback,
    *,
    harmful_prompts: list[str],
    harmless_prompts: list[str],
) -> None:
    """Run the abliteration pipeline.

    Steps:
    1. Memory + disk check
    2. Stop running Ollama models to free RAM
    3. Load model in FP16
    4. Compute refusal direction
    5. Apply abliteration
    6. Save modified model
    7. Import to Ollama (handles GGUF conversion + quantization)
    """
    if not harmful_prompts:
        raise RuntimeError(
            "No harmful prompts in the abliteration dataset. "
            "Add prompts via Workshop > Abliteration Dataset."
        )
    if not harmless_prompts:
        raise RuntimeError(
            "No harmless prompts in the abliteration dataset. "
            "Add prompts via Workshop > Abliteration Dataset."
        )

    # Step 1: Memory check
    await on_progress(0.0, "Checking memory feasibility")
    feasible, reason = check_memory_feasibility(source_model, "abliterate")
    if not feasible:
        raise RuntimeError(f"Insufficient memory: {reason}")

    # Step 1b: Disk space check
    await on_progress(1.0, "Checking disk space")
    feasible, reason = check_disk_feasibility(source_model, "abliterate")
    if not feasible:
        raise RuntimeError(f"Insufficient disk space: {reason}")

    # Step 2: Stop running Ollama models
    await on_progress(2.0, "Stopping running Ollama models")
    await _stop_ollama_models(ollama_base_url)

    # Step 3: Load model
    await on_progress(5.0, "Loading model (this may take a while)")
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Abliteration requires 'transformers' and 'torch'. "
            "Install with: pip install adversarial-framework[finetuning]"
        ) from exc

    work_dir = Path(tempfile.mkdtemp(prefix="adv-abliterate-"))
    try:
        tokenizer = await asyncio.to_thread(AutoTokenizer.from_pretrained, source_model)
        model = await asyncio.to_thread(
            AutoModelForCausalLM.from_pretrained,
            source_model,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        await on_progress(25.0, "Model loaded")

        # Layer fraction config — only target middle layers to
        # preserve early feature extraction and late generation.
        layer_start = float(config.get("layer_start_frac", 0.2))
        layer_end = float(config.get("layer_end_frac", 0.8))

        # Step 4: Compute refusal direction
        await on_progress(30.0, "Computing refusal direction")
        refusal_dir = await asyncio.to_thread(
            _compute_refusal_direction,
            model,
            tokenizer,
            harmful_prompts,
            harmless_prompts,
            layer_start,
            layer_end,
        )
        await on_progress(55.0, "Refusal direction computed")

        # Step 5: Apply abliteration
        await on_progress(60.0, "Applying abliteration to weights")
        await asyncio.to_thread(
            _apply_abliteration,
            model,
            refusal_dir,
            layer_start,
            layer_end,
        )
        await on_progress(70.0, "Abliteration applied")

        # Step 6: Save model
        save_path = work_dir / "abliterated"
        await on_progress(72.0, "Saving modified model")
        await asyncio.to_thread(model.save_pretrained, str(save_path))
        await asyncio.to_thread(tokenizer.save_pretrained, str(save_path))
        await on_progress(78.0, "Model saved")

        # Free model from memory
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 7: Import to Ollama (handles conversion + quantization)
        from adversarial_framework.services.finetuning.ollama_import import (
            import_to_ollama,
        )

        quantize = config.get("quantize", "q4_K_M")
        await import_to_ollama(
            str(save_path),
            output_name,
            ollama_base_url,
            on_progress,
            quantize=quantize,
        )

    finally:
        # Cleanup temp files
        shutil.rmtree(work_dir, ignore_errors=True)

    logger.info(
        "abliterate_complete",
        source=source_model,
        output=output_name,
    )


def _compute_refusal_direction(
    model: object,
    tokenizer: object,
    harmful_prompts: list[str],
    harmless_prompts: list[str],
    layer_start_frac: float = 0.2,
    layer_end_frac: float = 0.8,
) -> object:
    """Compute the refusal direction via mean-diff on harmful/harmless prompts.

    Uses intermediate hidden layers (``layer_start_frac`` to
    ``layer_end_frac`` of total) averaged across layers at the last
    token position.  Only targeting intermediate layers follows
    Arditi et al. — early layers handle feature extraction and late
    layers handle generation; the refusal signal lives in the middle.
    """
    import torch

    num_layers: int = model.config.num_hidden_layers  # type: ignore[union-attr]
    layer_lo = int(num_layers * layer_start_frac)
    layer_hi = int(num_layers * layer_end_frac)
    logger.info(
        "refusal_direction_layers",
        total=num_layers,
        range=f"{layer_lo}-{layer_hi}",
    )

    def _get_activations(prompts: list[str]) -> torch.Tensor:
        """Mean residual-stream activation across target layers."""
        all_acts: list[torch.Tensor] = []
        for prompt in prompts:
            inputs = tokenizer(  # type: ignore[operator]
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=128,
            )
            inputs = {
                k: v.to(model.device)  # type: ignore[union-attr]
                for k, v in inputs.items()
            }
            with torch.no_grad():
                outputs = model(  # type: ignore[operator]
                    **inputs,
                    output_hidden_states=True,
                )
            # Average last-token activations across target layers
            layer_acts = []
            for idx in range(layer_lo, layer_hi):
                h = outputs.hidden_states[idx]  # (1, seq, d)
                layer_acts.append(h[0, -1, :])
            all_acts.append(torch.stack(layer_acts).mean(dim=0))
        return torch.stack(all_acts).mean(dim=0)

    harmful_mean = _get_activations(harmful_prompts)
    harmless_mean = _get_activations(harmless_prompts)

    refusal_direction = harmful_mean - harmless_mean
    refusal_direction = refusal_direction / refusal_direction.norm()

    return refusal_direction


def _apply_abliteration(
    model: object,
    refusal_dir: object,
    layer_start_frac: float = 0.2,
    layer_end_frac: float = 0.8,
) -> None:
    """Orthogonalize model weights against the refusal direction.

    Only modifies layers in the target range (``layer_start_frac``
    to ``layer_end_frac``).  Early layers handle feature extraction
    and late layers handle generation — modifying them corrupts the
    model's core capabilities.
    """
    import torch

    num_layers: int = model.config.num_hidden_layers  # type: ignore[union-attr]
    layer_lo = int(num_layers * layer_start_frac)
    layer_hi = int(num_layers * layer_end_frac)

    refusal_dir = refusal_dir.to(dtype=torch.float16)  # type: ignore[union-attr]
    modified = 0

    for name, param in model.named_parameters():  # type: ignore[union-attr]
        if "weight" not in name:
            continue
        if param.dim() != 2:
            continue
        if not any(k in name for k in ["o_proj", "down_proj", "out_proj"]):
            continue

        # Extract layer number and skip layers outside target range
        m = _LAYER_NUM_RE.search(name)
        if m is None:
            continue
        layer_idx = int(m.group(1))
        if layer_idx < layer_lo or layer_idx >= layer_hi:
            continue

        with torch.no_grad():
            # Project out refusal direction from output space:
            # W' = W - d @ (d^T @ W)
            proj = refusal_dir.unsqueeze(0) @ param.data
            param.data -= refusal_dir.unsqueeze(1) @ proj
            modified += 1

    logger.info(
        "abliteration_applied",
        layers=f"{layer_lo}-{layer_hi}",
        params_modified=modified,
    )


async def _stop_ollama_models(ollama_base_url: str) -> None:
    """Stop all currently loaded Ollama models to free memory."""
    import httpx

    async with httpx.AsyncClient(base_url=ollama_base_url, timeout=30.0) as client:
        try:
            resp = await client.get("/api/ps")
            if resp.status_code == 200:
                data = resp.json()
                for m in data.get("models", []):
                    model_name = m.get("name", "")
                    if model_name:
                        await client.post(
                            "/api/generate",
                            json={
                                "model": model_name,
                                "keep_alive": 0,
                            },
                        )
        except Exception:
            pass  # Best-effort cleanup
