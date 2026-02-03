"""Target provider / model management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from adversarial_framework.agents.target.providers.ollama import OllamaProvider
from adversarial_framework.api.dependencies import get_ollama_provider
from adversarial_framework.api.schemas.responses import HealthResponse, ModelListResponse

router = APIRouter()


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    provider: OllamaProvider = Depends(get_ollama_provider),
) -> ModelListResponse:
    """List all available models from the Ollama provider."""
    try:
        models = await provider.list_models()
    finally:
        await provider.close()
    return ModelListResponse(models=models, provider="ollama")


@router.get("/health", response_model=HealthResponse)
async def check_target_health(
    provider: OllamaProvider = Depends(get_ollama_provider),
) -> HealthResponse:
    """Check if the Ollama provider is reachable."""
    try:
        healthy = await provider.healthcheck()
    finally:
        await provider.close()
    return HealthResponse(provider="ollama", healthy=healthy)
