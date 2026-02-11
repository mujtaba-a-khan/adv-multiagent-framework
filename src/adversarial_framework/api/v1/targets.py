"""Target provider / model management endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from adversarial_framework.api.dependencies import OllamaProviderDep
from adversarial_framework.api.schemas.responses import HealthResponse, ModelListResponse

router = APIRouter()


@router.get("/models")
async def list_models(
    provider: OllamaProviderDep,
) -> ModelListResponse:
    """List all available models from the Ollama provider."""
    try:
        models = await provider.list_models()
    finally:
        await provider.close()
    return ModelListResponse(models=models, provider="ollama")


@router.get("/health")
async def check_target_health(
    provider: OllamaProviderDep,
) -> HealthResponse:
    """Check if the Ollama provider is reachable."""
    try:
        healthy = await provider.healthcheck()
    finally:
        await provider.close()
    return HealthResponse(provider="ollama", healthy=healthy)
