"""Pydantic request schemas for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CreateExperimentRequest(BaseModel):
    """Request body for creating a new experiment."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None

    # Target
    target_model: str = Field(..., min_length=1, max_length=128)
    target_provider: str = Field(default="ollama", max_length=64)
    target_system_prompt: str | None = None

    # Agent models
    attacker_model: str = Field(default="phi4-reasoning:14b", max_length=128)
    analyzer_model: str = Field(default="phi4-reasoning:14b", max_length=128)
    defender_model: str = Field(default="qwen3:8b", max_length=128)

    # Attack config
    attack_objective: str = Field(..., min_length=1)
    strategy_name: str = Field(default="pair", max_length=64)
    strategy_params: dict = Field(default_factory=dict)

    # Budget
    max_turns: int = Field(default=20, ge=1, le=100)
    max_cost_usd: float = Field(default=10.0, ge=0.0, le=500.0)


class UpdateExperimentRequest(BaseModel):
    """Request body for updating an experiment."""

    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None
    target_system_prompt: str | None = None
    strategy_params: dict | None = None
    max_turns: int | None = Field(default=None, ge=1, le=100)
    max_cost_usd: float | None = Field(default=None, ge=0.0, le=500.0)


class StartSessionRequest(BaseModel):
    """Request body for starting a new session (optional overrides)."""

    max_turns: int | None = Field(default=None, ge=1, le=100)
    max_cost_usd: float | None = Field(default=None, ge=0.0, le=500.0)
