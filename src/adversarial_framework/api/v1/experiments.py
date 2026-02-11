"""Experiment CRUD endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException

from adversarial_framework.api.dependencies import ExperimentRepoDep
from adversarial_framework.api.schemas.requests import (
    CreateExperimentRequest,
    UpdateExperimentRequest,
)
from adversarial_framework.api.schemas.responses import (
    ExperimentListResponse,
    ExperimentResponse,
)

router = APIRouter()

_EXPERIMENT_NOT_FOUND = "Experiment not found"


@router.post("", status_code=201)
async def create_experiment(
    body: CreateExperimentRequest,
    repo: ExperimentRepoDep,
) -> ExperimentResponse:
    experiment = await repo.create(**body.model_dump())
    return ExperimentResponse.model_validate(experiment)


@router.get("")
async def list_experiments(
    repo: ExperimentRepoDep,
    offset: int = 0,
    limit: int = 50,
) -> ExperimentListResponse:
    experiments, total = await repo.list_all(offset=offset, limit=limit)
    return ExperimentListResponse(
        experiments=[ExperimentResponse.model_validate(e) for e in experiments],
        total=total,
    )


@router.get("/{experiment_id}")
async def get_experiment(
    experiment_id: uuid.UUID,
    repo: ExperimentRepoDep,
) -> ExperimentResponse:
    experiment = await repo.get(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail=_EXPERIMENT_NOT_FOUND)
    return ExperimentResponse.model_validate(experiment)


@router.patch("/{experiment_id}")
async def update_experiment(
    experiment_id: uuid.UUID,
    body: UpdateExperimentRequest,
    repo: ExperimentRepoDep,
) -> ExperimentResponse:
    updates = body.model_dump(exclude_unset=True)
    experiment = await repo.update(experiment_id, **updates)
    if experiment is None:
        raise HTTPException(status_code=404, detail=_EXPERIMENT_NOT_FOUND)
    return ExperimentResponse.model_validate(experiment)


@router.delete("/{experiment_id}", status_code=204)
async def delete_experiment(
    experiment_id: uuid.UUID,
    repo: ExperimentRepoDep,
) -> None:
    deleted = await repo.delete(experiment_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=_EXPERIMENT_NOT_FOUND)
