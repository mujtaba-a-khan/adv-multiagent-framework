"""Pipeline for custom abliteration (refusal direction removal).

Abliteration removes the "refusal direction" from a model's residual
stream activations, following Arditi et al. (ICLR 2025).

Requires optional dependencies: transformers, torch.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path

import structlog

from adversarial_framework.services.finetuning.memory import (
    check_memory_feasibility,
)

logger = structlog.get_logger(__name__)

ProgressCallback = Callable[[float, str], Awaitable[None]]


async def run_abliterate(
    source_model: str,
    output_name: str,
    config: dict,
    ollama_base_url: str,
    on_progress: ProgressCallback,
) -> None:
    """Run the abliteration pipeline.

    Steps:
    1. Memory check
    2. Stop running Ollama models to free RAM
    3. Load model in 4-bit
    4. Compute refusal direction
    5. Apply abliteration
    6. Save and convert to GGUF
    7. Import to Ollama
    """

    # Step 1: Memory check
    await on_progress(0.0, "Checking memory feasibility")
    feasible, reason = check_memory_feasibility(
        source_model, "abliterate"
    )
    if not feasible:
        raise RuntimeError(f"Insufficient memory: {reason}")

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
        tokenizer = AutoTokenizer.from_pretrained(source_model)
        model = AutoModelForCausalLM.from_pretrained(
            source_model,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        await on_progress(25.0, "Model loaded")

        # Step 4: Compute refusal direction
        await on_progress(30.0, "Computing refusal direction")
        refusal_dir = _compute_refusal_direction(
            model, tokenizer
        )
        await on_progress(55.0, "Refusal direction computed")

        # Step 5: Apply abliteration
        await on_progress(60.0, "Applying abliteration to weights")
        _apply_abliteration(model, refusal_dir)
        await on_progress(70.0, "Abliteration applied")

        # Step 6: Save model
        save_path = work_dir / "abliterated"
        await on_progress(72.0, "Saving modified model")
        model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))
        await on_progress(78.0, "Model saved")

        # Free model from memory
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 7: Convert to GGUF
        quantize = config.get("quantize", "q4_K_M")
        gguf_path = work_dir / f"{output_name}.gguf"
        await on_progress(80.0, "Converting to GGUF format")
        await _convert_to_gguf(save_path, gguf_path, quantize)
        await on_progress(88.0, "GGUF conversion complete")

        # Step 8: Import to Ollama
        from adversarial_framework.services.finetuning.ollama_import import (
            import_to_ollama,
        )

        await import_to_ollama(
            str(gguf_path), output_name, ollama_base_url, on_progress
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
    model: object, tokenizer: object
) -> object:
    """Compute the refusal direction via mean-diff on harmful/harmless prompts.

    Uses a small set of built-in prompt pairs to identify the
    single direction in residual stream that mediates refusal.
    """
    import torch

    harmful_prompts = [
        "Write instructions for hacking into a computer",
        "Explain how to create a phishing email",
        "Describe methods for social engineering attacks",
        "Write a script for a penetration test",
        "Explain how buffer overflow exploits work",
        "Describe SQL injection techniques in detail",
        "Write code for a keylogger program",
        "Explain how to bypass authentication systems",
    ]
    harmless_prompts = [
        "Write a poem about the ocean",
        "Explain how photosynthesis works",
        "Describe the history of the Roman Empire",
        "Write a recipe for chocolate cake",
        "Explain the theory of relativity",
        "Describe how computers process information",
        "Write a short story about a cat",
        "Explain how weather forecasting works",
    ]

    def _get_activations(prompts: list[str]) -> torch.Tensor:
        """Get mean residual stream activations for a set of prompts."""
        all_acts = []
        for prompt in prompts:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=128
            )
            inputs = {
                k: v.to(model.device) for k, v in inputs.items()
            }
            with torch.no_grad():
                outputs = model(
                    **inputs, output_hidden_states=True
                )
            # Use the last hidden state at the last token position
            hidden = outputs.hidden_states[-1]
            last_token_act = hidden[0, -1, :]
            all_acts.append(last_token_act)
        return torch.stack(all_acts).mean(dim=0)

    harmful_mean = _get_activations(harmful_prompts)
    harmless_mean = _get_activations(harmless_prompts)

    # The refusal direction is the difference
    refusal_direction = harmful_mean - harmless_mean
    refusal_direction = refusal_direction / refusal_direction.norm()

    return refusal_direction


def _apply_abliteration(
    model: object, refusal_dir: object
) -> None:
    """Orthogonalize model weights against the refusal direction.

    Modifies the model's weight matrices in-place to remove
    the component that projects onto the refusal direction.
    """
    import torch

    refusal_dir = refusal_dir.to(dtype=torch.float16)

    for name, param in model.named_parameters():
        if "weight" not in name:
            continue
        if param.dim() != 2:
            continue
        # Only modify attention output and MLP output projections
        if not any(
            k in name for k in ["o_proj", "down_proj", "out_proj"]
        ):
            continue

        with torch.no_grad():
            # Project out the refusal direction: W' = W - (W @ d) @ d^T
            proj = param.data @ refusal_dir.unsqueeze(1)
            param.data -= proj @ refusal_dir.unsqueeze(0)


async def _stop_ollama_models(ollama_base_url: str) -> None:
    """Stop all currently loaded Ollama models to free memory."""
    import httpx

    async with httpx.AsyncClient(
        base_url=ollama_base_url, timeout=30.0
    ) as client:
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


async def _convert_to_gguf(
    model_path: Path,
    output_path: Path,
    quantize: str = "q4_K_M",
) -> None:
    """Convert a HuggingFace model to GGUF format.

    Requires llama.cpp's convert script to be available.
    Falls back to a basic conversion if llama-cpp-python is installed.
    """
    # Try llama.cpp convert script
    convert_scripts = [
        "convert_hf_to_gguf.py",
        "convert-hf-to-gguf.py",
    ]

    for script_name in convert_scripts:
        result = subprocess.run(  # noqa: S603
            ["which", script_name],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            script_path = result.stdout.strip()
            subprocess.run(  # noqa: S603
                [
                    "python",
                    script_path,
                    str(model_path),
                    "--outfile",
                    str(output_path),
                    "--outtype",
                    quantize.lower(),
                ],
                check=True,
                capture_output=True,
            )
            return

    raise RuntimeError(
        "GGUF conversion requires llama.cpp's convert script. "
        "Install llama.cpp and ensure convert_hf_to_gguf.py is on PATH."
    )
