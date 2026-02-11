"""V1 API router — aggregates all endpoint sub-routers."""

from __future__ import annotations

from fastapi import APIRouter

from adversarial_framework.api.v1.comparisons import router as comparisons_router
from adversarial_framework.api.v1.dataset import router as dataset_router
from adversarial_framework.api.v1.defenses import router as defenses_router
from adversarial_framework.api.v1.experiments import router as experiments_router
from adversarial_framework.api.v1.finetuning import router as finetuning_router
from adversarial_framework.api.v1.playground import router as playground_router
from adversarial_framework.api.v1.sessions import router as sessions_router
from adversarial_framework.api.v1.strategies import router as strategies_router
from adversarial_framework.api.v1.targets import router as targets_router
from adversarial_framework.api.v1.turns import router as turns_router
from adversarial_framework.api.v1.ws import router as ws_router

v1_router = APIRouter()

_EXPERIMENTS_PREFIX = "/experiments"

v1_router.include_router(experiments_router, prefix=_EXPERIMENTS_PREFIX, tags=["Experiments"])
v1_router.include_router(sessions_router, prefix=_EXPERIMENTS_PREFIX, tags=["Sessions"])
v1_router.include_router(comparisons_router, prefix=_EXPERIMENTS_PREFIX, tags=["Comparisons"])
v1_router.include_router(turns_router, prefix="/sessions", tags=["Turns"])
v1_router.include_router(strategies_router, prefix="/strategies", tags=["Strategies"])
v1_router.include_router(defenses_router, prefix="/defenses", tags=["Defenses"])
v1_router.include_router(targets_router, prefix="/targets", tags=["Targets"])
v1_router.include_router(finetuning_router, prefix="/finetuning", tags=["Fine-Tuning"])
v1_router.include_router(
    dataset_router, prefix="/finetuning/dataset", tags=["Abliteration Dataset"]
)
v1_router.include_router(
    playground_router, prefix="/playground", tags=["Playground"]
)
v1_router.include_router(ws_router, tags=["WebSocket"])
