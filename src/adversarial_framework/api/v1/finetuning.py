"""Fine-tuning endpoints for create, list, start, cancel, delete jobs."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from adversarial_framework.agents.target.providers.ollama import (
    OllamaProvider,
)
from adversarial_framework.api.dependencies import (
    get_finetuning_repo,
    get_ollama_provider,
)
from adversarial_framework.api.schemas.requests import (
    CreateFineTuningJobRequest,
)
from adversarial_framework.api.schemas.responses import (
    FineTuningJobListResponse,
    FineTuningJobResponse,
)
from adversarial_framework.db.repositories.finetuning import (
    FineTuningJobRepository,
)

logger = structlog.get_logger(__name__)
router = APIRouter()


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


@router.post("/jobs", response_model=FineTuningJobResponse, status_code=201)
async def create_job(
    body: CreateFineTuningJobRequest,
    repo: FineTuningJobRepository = Depends(get_finetuning_repo),
) -> FineTuningJobResponse:
    job = await repo.create(
        name=body.name,
        job_type=body.job_type,
        source_model=body.source_model,
        output_model_name=body.output_model_name,
        config=body.config,
    )
    return FineTuningJobResponse.model_validate(job)


@router.get("/jobs", response_model=FineTuningJobListResponse)
async def list_jobs(
    offset: int = 0,
    limit: int = 50,
    status: str | None = None,
    repo: FineTuningJobRepository = Depends(get_finetuning_repo),
) -> FineTuningJobListResponse:
    jobs, total = await repo.list_all(
        offset=offset, limit=limit, status=status
    )
    return FineTuningJobListResponse(
        jobs=[FineTuningJobResponse.model_validate(j) for j in jobs],
        total=total,
    )


@router.get("/jobs/{job_id}", response_model=FineTuningJobResponse)
async def get_job(
    job_id: uuid.UUID,
    repo: FineTuningJobRepository = Depends(get_finetuning_repo),
) -> FineTuningJobResponse:
    job = await repo.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return FineTuningJobResponse.model_validate(job)


@router.delete("/jobs/{job_id}", status_code=204)
async def delete_job(
    job_id: uuid.UUID,
    repo: FineTuningJobRepository = Depends(get_finetuning_repo),
) -> None:
    job = await repo.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == "running":
        raise HTTPException(
            status_code=409, detail="Cannot delete a running job"
        )
    await repo.delete(job_id)


#Start / Cancel


@router.post(
    "/jobs/{job_id}/start", response_model=FineTuningJobResponse
)
async def start_job(
    job_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    repo: FineTuningJobRepository = Depends(get_finetuning_repo),
    provider: OllamaProvider = Depends(get_ollama_provider),
) -> FineTuningJobResponse:
    """Start a fine-tuning/abliteration job in the background."""
    job = await repo.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "pending":
        raise HTTPException(
            status_code=409,
            detail=f"Job is already '{job.status}'",
        )

    # Snapshot BEFORE update_status (which calls expire_all)
    snapshot = _FineTuningJobSnapshot.from_orm(job)
    ollama_base_url = provider._base_url

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
    "/jobs/{job_id}/cancel", response_model=FineTuningJobResponse
)
async def cancel_job(
    job_id: uuid.UUID,
    repo: FineTuningJobRepository = Depends(get_finetuning_repo),
) -> FineTuningJobResponse:
    """Request cancellation of a running job."""
    from adversarial_framework.services.finetuning.runner import (
        request_cancellation,
    )

    job = await repo.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "running":
        raise HTTPException(
            status_code=409, detail="Job is not running"
        )

    request_cancellation(str(job_id))
    return FineTuningJobResponse.model_validate(job)


# Model Management


@router.get("/models")
async def list_custom_models(
    provider: OllamaProvider = Depends(get_ollama_provider),
) -> dict:
    """List all models available in Ollama."""
    models = await provider.list_models()
    return {"models": models, "provider": "ollama"}


@router.delete("/models/{model_name}", status_code=204)
async def delete_model(
    model_name: str,
    provider: OllamaProvider = Depends(get_ollama_provider),
) -> None:
    """Delete a model from Ollama."""
    try:
        await provider._client.delete(
            "/api/delete", json={"name": model_name}
        )
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
