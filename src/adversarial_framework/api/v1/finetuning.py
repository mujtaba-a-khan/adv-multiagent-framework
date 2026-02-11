"""Fine-tuning endpoints for create, list, start, cancel, delete jobs."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass

import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException

from adversarial_framework.api.dependencies import (
    FineTuningRepoDep,
    OllamaProviderDep,
)
from adversarial_framework.api.schemas.requests import (
    CreateFineTuningJobRequest,
)
from adversarial_framework.api.schemas.responses import (
    FineTuningJobListResponse,
    FineTuningJobResponse,
)

logger = structlog.get_logger(__name__)
router = APIRouter()

_JOB_NOT_FOUND = "Job not found"


@dataclass(frozen=True)
class _FineTuningJobSnapshot:
    """Plain-data copy of a FineTuningJob ORM object.

    Background tasks must not hold references to ORM objects.
    """

    id: uuid.UUID
    name: str
    job_type: str
    source_model: str
    output_model_name: str
    config: dict

    @classmethod
    def from_orm(cls, job: object) -> _FineTuningJobSnapshot:
        return cls(
            id=job.id,
            name=job.name,
            job_type=job.job_type,
            source_model=job.source_model,
            output_model_name=job.output_model_name,
            config=dict(job.config) if job.config else {},
        )


# CRUD


@router.post("/jobs", status_code=201)
async def create_job(
    body: CreateFineTuningJobRequest,
    repo: FineTuningRepoDep,
) -> FineTuningJobResponse:
    job = await repo.create(
        name=body.name,
        job_type=body.job_type,
        source_model=body.source_model,
        output_model_name=body.output_model_name,
        config=body.config,
    )
    return FineTuningJobResponse.model_validate(job)


@router.get("/jobs")
async def list_jobs(
    repo: FineTuningRepoDep,
    offset: int = 0,
    limit: int = 50,
    status: str | None = None,
) -> FineTuningJobListResponse:
    jobs, total = await repo.list_all(offset=offset, limit=limit, status=status)
    return FineTuningJobListResponse(
        jobs=[FineTuningJobResponse.model_validate(j) for j in jobs],
        total=total,
    )


@router.get(
    "/jobs/{job_id}",
    responses={404: {"description": "Job not found"}},
)
async def get_job(
    job_id: uuid.UUID,
    repo: FineTuningRepoDep,
) -> FineTuningJobResponse:
    job = await repo.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=_JOB_NOT_FOUND)
    return FineTuningJobResponse.model_validate(job)


@router.delete(
    "/jobs/{job_id}",
    status_code=204,
    responses={
        404: {"description": "Job not found"},
        409: {"description": "Cannot delete a running job"},
    },
)
async def delete_job(
    job_id: uuid.UUID,
    repo: FineTuningRepoDep,
) -> None:
    job = await repo.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=_JOB_NOT_FOUND)
    if job.status == "running":
        raise HTTPException(status_code=409, detail="Cannot delete a running job")
    await repo.delete(job_id)


# Start / Cancel


@router.post(
    "/jobs/{job_id}/start",
    responses={
        404: {"description": "Job not found"},
        409: {"description": "Job already started or completed"},
    },
)
async def start_job(
    job_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    repo: FineTuningRepoDep,
    provider: OllamaProviderDep,
) -> FineTuningJobResponse:
    """Start a fine-tuning/abliteration job in the background."""
    job = await repo.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=_JOB_NOT_FOUND)
    if job.status != "pending":
        raise HTTPException(
            status_code=409,
            detail=f"Job is already '{job.status}'",
        )

    # Snapshot BEFORE update_status (which calls expire_all)
    snapshot = _FineTuningJobSnapshot.from_orm(job)
    ollama_base_url = provider.base_url

    await repo.update_status(job_id, "running")
    await repo.db.commit()

    background_tasks.add_task(
        _run_finetuning_job,
        snapshot=snapshot,
        ollama_base_url=ollama_base_url,
    )

    job = await repo.get(job_id)
    return FineTuningJobResponse.model_validate(job)


@router.post(
    "/jobs/{job_id}/cancel",
    responses={
        404: {"description": "Job not found"},
        409: {"description": "Job is not running"},
    },
)
async def cancel_job(
    job_id: uuid.UUID,
    repo: FineTuningRepoDep,
) -> FineTuningJobResponse:
    """Request cancellation of a running job."""
    from adversarial_framework.services.finetuning.runner import (
        request_cancellation,
    )

    job = await repo.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=_JOB_NOT_FOUND)
    if job.status != "running":
        raise HTTPException(status_code=409, detail="Job is not running")

    request_cancellation(str(job_id))
    return FineTuningJobResponse.model_validate(job)


# Disk Management


@router.get("/disk-status")
async def disk_status() -> dict:
    """Return disk usage summary including orphan blob detection."""
    from adversarial_framework.services.finetuning.ollama_import import (
        get_disk_status,
    )

    return await asyncio.to_thread(get_disk_status)


@router.post("/cleanup-orphans")
async def cleanup_orphans() -> dict:
    """Delete orphaned Ollama blobs not referenced by any model."""
    from adversarial_framework.services.finetuning.ollama_import import (
        cleanup_orphan_blobs,
    )

    return await asyncio.to_thread(cleanup_orphan_blobs)


# Model Management


@router.get("/models")
async def list_custom_models(
    provider: OllamaProviderDep,
) -> dict:
    """List all models available in Ollama."""
    models = await provider.list_models()
    return {"models": models, "provider": "ollama"}


@router.delete(
    "/models/{model_name}",
    status_code=204,
    responses={500: {"description": "Failed to delete model"}},
)
async def delete_model(
    model_name: str,
    provider: OllamaProviderDep,
) -> None:
    """Delete a model from Ollama."""
    try:
        await provider._client.delete("/api/delete", json={"name": model_name})
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete model: {exc}",
        ) from exc


# Background runner


async def _run_finetuning_job(
    snapshot: _FineTuningJobSnapshot,
    ollama_base_url: str,
) -> None:
    """Execute a fine-tuning job with real-time progress streaming."""
    from adversarial_framework.services.finetuning.runner import (
        run_job,
    )

    await run_job(snapshot, ollama_base_url)
