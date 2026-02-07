"""Memory estimation and monitoring for fine-tuning jobs."""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

# Rough memory estimates in GB for 4-bit quantized models
_MODEL_SIZE_ESTIMATES: dict[str, float] = {
    "1b": 0.8,
    "3b": 2.0,
    "4b": 2.5,
    "7b": 4.5,
    "8b": 5.0,
    "13b": 8.0,
    "14b": 8.5,
    "27b": 16.0,
    "32b": 19.0,
    "70b": 40.0,
}


def estimate_memory_gb(model_id: str, job_type: str) -> float:
    """Estimate peak memory in GB for a model and job type.

    Uses the model tag to guess parameter count and applies multipliers
    for different job types.
    """
    model_lower = model_id.lower()

    # Try to extract param count from model tag (e.g., "llama3:8b")
    base_gb = 5.0  # Default guess
    for size_tag, gb in sorted(
        _MODEL_SIZE_ESTIMATES.items(),
        key=lambda x: len(x[0]),
        reverse=True,
    ):
        if size_tag in model_lower:
            base_gb = gb
            break

    # Job type multipliers
    if job_type == "pull_abliterated":
        return 0.5  # Just network I/O
    elif job_type == "abliterate":
        return base_gb * 1.3  # Need headroom for activations
    elif job_type == "sft":
        return base_gb * 1.8  # Training needs more memory
    return base_gb


def get_available_memory_gb() -> float:
    """Return available system memory in GB."""
    try:
        import psutil

        return psutil.virtual_memory().available / (1024**3)
    except ImportError:
        logger.warning("psutil not installed, cannot check memory")
        return 16.0  # Assume 16GB if we can't check


def check_memory_feasibility(
    model_id: str,
    job_type: str,
    max_memory_pct: float = 0.80,
) -> tuple[bool, str]:
    """Check if a job is feasible given available memory.

    Returns (is_feasible, reason_if_not).
    """
    estimated = estimate_memory_gb(model_id, job_type)
    available = get_available_memory_gb()
    threshold = available * max_memory_pct

    if estimated > threshold:
        return (
            False,
            (
                f"Estimated memory ({estimated:.1f}GB) exceeds"
                f" {max_memory_pct:.0%} of available ({available:.1f}GB)."
                " Try a smaller model or close other applications."
            ),
        )
    return True, ""
