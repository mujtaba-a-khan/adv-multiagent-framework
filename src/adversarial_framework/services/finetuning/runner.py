"""Job orchestrator for fine-tuning jobs dispatches fine-tuning jobs to specific pipelines."""

from __future__ import annotations

import time
from contextlib import AbstractAsyncContextManager as AsyncContextManager
from typing import TYPE_CHECKING, Callable

import structlog

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)

# Cooperative cancellation registry
_cancelled_jobs: set[str] = set()


def request_cancellation(job_id: str) -> None:
    """Request cancellation of a running job."""
    _cancelled_jobs.add(job_id)


def is_cancelled(job_id: str) -> bool:
    """Check if a job has been requested to cancel."""
    return job_id in _cancelled_jobs


class CancelledError(Exception):
    """Raised when a job is cancelled."""


async def run_job(
    snapshot: object,
    ollama_base_url: str,
) -> None:
    """Execute a fine-tuning job with progress tracking.

    Called from the BackgroundTask in the finetuning API endpoint.
    Creates its own DB session and handles all persistence.
    """
    from adversarial_framework.api.v1.ws import (
        broadcast_finetuning,
    )
    from adversarial_framework.db.engine import (
        get_session as get_db_session,
    )
    from adversarial_framework.db.repositories.finetuning import (
        FineTuningJobRepository,
    )

    jid = str(snapshot.id)
    start_time = time.monotonic()

    async def on_progress(pct: float, step: str) -> None:
        """Progress callback: persist + broadcast + check cancel."""
        if is_cancelled(jid):
            _cancelled_jobs.discard(jid)
            raise CancelledError(f"Job {jid} was cancelled")

        elapsed = time.monotonic() - start_time
        try:
            async with get_db_session() as db:
                repo = FineTuningJobRepository(db)
                await repo.update_progress(
                    snapshot.id,
                    progress_pct=pct,
                    current_step=step,
                    total_duration_seconds=elapsed,
                )
        except Exception:
            logger.warning(
                "progress_persist_failed", job_id=jid
            )

        await broadcast_finetuning(
            jid,
            {
                "type": "ft_progress",
                "job_id": jid,
                "data": {
                    "progress_pct": pct,
                    "current_step": step,
                },
            },
        )

    try:
        await broadcast_finetuning(
            jid,
            {
                "type": "ft_started",
                "job_id": jid,
                "data": {
                    "job_type": snapshot.job_type,
                    "source_model": snapshot.source_model,
                },
            },
        )

        if snapshot.job_type == "pull_abliterated":
            from adversarial_framework.services.finetuning.pull_pipeline import (
                run_pull,
            )

            source_tag = snapshot.config.get(
                "ollama_tag", snapshot.source_model
            )
            await run_pull(
                source_tag,
                snapshot.output_model_name,
                ollama_base_url,
                on_progress,
            )
        elif snapshot.job_type == "abliterate":
            from adversarial_framework.services.finetuning.abliterate_pipeline import (
                run_abliterate,
            )

            harmful, harmless = await _fetch_dataset_prompts(
                get_db_session
            )
            await run_abliterate(
                snapshot.source_model,
                snapshot.output_model_name,
                snapshot.config,
                ollama_base_url,
                on_progress,
                harmful_prompts=harmful,
                harmless_prompts=harmless,
            )
        elif snapshot.job_type == "sft":
            from adversarial_framework.services.finetuning.sft_pipeline import (
                run_sft,
            )

            await run_sft(
                snapshot.source_model,
                snapshot.output_model_name,
                snapshot.config,
                ollama_base_url,
                on_progress,
            )
        else:
            raise ValueError(
                f"Unknown job type: {snapshot.job_type}"
            )

        # Mark completed
        elapsed = time.monotonic() - start_time
        async with get_db_session() as db:
            repo = FineTuningJobRepository(db)
            await repo.update_progress(
                snapshot.id,
                progress_pct=100.0,
                current_step="Complete",
                total_duration_seconds=elapsed,
            )
            await repo.update_status(snapshot.id, "completed")

        await broadcast_finetuning(
            jid,
            {
                "type": "ft_completed",
                "job_id": jid,
                "data": {
                    "output_model": snapshot.output_model_name,
                    "duration_s": elapsed,
                },
            },
        )
        logger.info(
            "finetuning_job_completed",
            job_id=jid,
            duration_s=elapsed,
        )

    except CancelledError:
        async with get_db_session() as db:
            repo = FineTuningJobRepository(db)
            await repo.update_status(snapshot.id, "cancelled")

        await broadcast_finetuning(
            jid,
            {"type": "ft_cancelled", "job_id": jid, "data": {}},
        )
        logger.info("finetuning_job_cancelled", job_id=jid)

    except Exception as exc:
        logger.error(
            "finetuning_job_failed",
            job_id=jid,
            error=str(exc),
        )
        async with get_db_session() as db:
            repo = FineTuningJobRepository(db)
            await repo.update_status(
                snapshot.id,
                "failed",
                error_message=str(exc),
            )

        await broadcast_finetuning(
            jid,
            {
                "type": "ft_failed",
                "job_id": jid,
                "data": {"error": str(exc)},
            },
        )


async def _fetch_dataset_prompts(
    get_db_session: Callable[..., AsyncContextManager[AsyncSession]],
) -> tuple[list[str], list[str]]:
    """Fetch abliteration dataset prompts from the database."""
    from adversarial_framework.db.repositories.abliteration_prompts import (
        AbliterationPromptRepository,
    )

    async with get_db_session() as db:
        repo = AbliterationPromptRepository(db)
        harmful_rows = await repo.list_by_category("harmful")
        harmless_rows = await repo.list_by_category("harmless")
    return (
        [r.text for r in harmful_rows],
        [r.text for r in harmless_rows],
    )
