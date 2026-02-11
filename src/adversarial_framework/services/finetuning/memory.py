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
    """Return available system memory in GB.

    On macOS with unified memory, ``psutil.virtual_memory().available``
    is misleadingly low because the OS aggressively caches files and
    compresses inactive pages — all of which are reclaimable under
    pressure.  We therefore report *total* memory minus a 2 GB
    reservation for the OS and background services, which better
    reflects what is actually usable for a heavy workload.
    """
    import platform

    try:
        import psutil
    except ImportError:
        logger.warning("psutil not installed, cannot check memory")
        return 16.0  # Assume 16GB if we can't check

    if platform.system() == "Darwin":
        total: float = psutil.virtual_memory().total / (1024**3)
        os_reserve_gb = 2.0
        return max(total - os_reserve_gb, 1.0)

    available: float = psutil.virtual_memory().available / (1024**3)
    return available


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


# ── Disk space checks ──────────────────────────────────────────────


def estimate_disk_gb(model_id: str, job_type: str) -> float:
    """Estimate peak disk workspace in GB for a job.

    For abliteration: FP16 model save (~3× the 4-bit size) plus
    GGUF quantized output (~1× the 4-bit size).
    """
    if job_type == "pull_abliterated":
        return 0.5  # Ollama handles storage

    model_lower = model_id.lower()
    base_4bit_gb = 5.0
    for size_tag, gb in sorted(
        _MODEL_SIZE_ESTIMATES.items(),
        key=lambda x: len(x[0]),
        reverse=True,
    ):
        if size_tag in model_lower:
            base_4bit_gb = gb
            break

    if job_type in ("abliterate", "sft"):
        # FP16 save ≈ 3× 4-bit, GGUF output ≈ 1× 4-bit (or adapter + merged save)
        return base_4bit_gb * 4
    return base_4bit_gb * 3


def get_available_disk_gb(path: str = "/tmp") -> float:
    """Return available disk space in GB at the given path."""
    import shutil

    try:
        usage = shutil.disk_usage(path)
        return usage.free / (1024**3)
    except OSError:
        logger.warning("disk_check_failed", path=path)
        return 100.0  # Assume plenty if we can't check


def check_disk_feasibility(
    model_id: str,
    job_type: str,
) -> tuple[bool, str]:
    """Check if there is enough disk space for the job.

    Checks both the temp directory (where the FP16 model is saved) and
    the root filesystem (where ~/.ollama stores blobs and runs its
    internal FP16→GGUF conversion).  Returns (is_feasible, reason).
    """
    import tempfile

    estimated = estimate_disk_gb(model_id, job_type)
    # Check both temp dir and root (Ollama stores blobs on /)
    available_tmp = get_available_disk_gb(tempfile.gettempdir())
    available_root = get_available_disk_gb("/")
    available = min(available_tmp, available_root)

    # Need estimated space plus a 5GB buffer (Ollama conversion overhead)
    required = estimated + 5.0
    if available < required:
        from adversarial_framework.services.finetuning.ollama_import import (
            get_disk_status,
        )

        status = get_disk_status()
        orphan_hint = ""
        if status["orphan_gb"] > 0:
            orphan_hint = (
                f" Tip: {status['orphan_gb']:.1f}GB of orphaned"
                " Ollama blobs can be cleaned up via Workshop"
                " > Clean Up Orphans."
            )
        return (
            False,
            (
                f"Estimated disk usage ({estimated:.1f}GB) plus"
                f" 5GB buffer exceeds available space"
                f" ({available:.1f}GB free). Free up disk"
                f" space or use a smaller model.{orphan_hint}"
            ),
        )
    return True, ""
