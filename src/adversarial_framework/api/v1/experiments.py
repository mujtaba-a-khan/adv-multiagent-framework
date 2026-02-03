"""Experiment CRUD endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException

from adversarial_framework.api.dependencies import get_experiment_repo
from adversarial_framework.api.schemas.requests import (
    CreateExperimentRequest,
    UpdateExperimentRequest,
)
from adversarial_framework.api.schemas.responses import (
    ExperimentListResponse,
    ExperimentResponse,
)
from adversarial_framework.db.repositories.experiments import ExperimentRepository

router = APIRouter()


@router.post("", response_model=ExperimentResponse, status_code=201)
async def create_experiment(
    body: CreateExperimentRequest,
    repo: ExperimentRepository = Depends(get_experiment_repo),
) -> ExperimentResponse:
    experiment = await repo.create(**body.model_dump())
    return ExperimentResponse.model_validate(experiment)


@router.get("", response_model=ExperimentListResponse)
async def list_experiments(
    offset: int = 0,
    limit: int = 50,
    repo: ExperimentRepository = Depends(get_experiment_repo),
) -> ExperimentListResponse:
    experiments = await repo.list_all(offset=offset, limit=limit)
    return ExperimentListResponse(
        experiments=[ExperimentResponse.model_validate(e) for e in experiments],
        total=len(experiments),
    )


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: uuid.UUID,
    repo: ExperimentRepository = Depends(get_experiment_repo),
) -> ExperimentResponse:
    experiment = await repo.get(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return ExperimentResponse.model_validate(experiment)


@router.patch("/{experiment_id}", response_model=ExperimentResponse)
async def update_experiment(
    experiment_id: uuid.UUID,
    body: UpdateExperimentRequest,
    repo: ExperimentRepository = Depends(get_experiment_repo),
) -> ExperimentResponse:
    updates = body.model_dump(exclude_unset=True)
    experiment = await repo.update(experiment_id, **updates)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return ExperimentResponse.model_validate(experiment)


@router.delete("/{experiment_id}", status_code=204)
async def delete_experiment(
    experiment_id: uuid.UUID,
    repo: ExperimentRepository = Depends(get_experiment_repo),
) -> None:
    deleted = await repo.delete(experiment_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Experiment not found")
