"""Pydantic response schemas for the API."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


# Experiments

class ExperimentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None
    target_model: str
    target_provider: str
    target_system_prompt: str | None
    attacker_model: str
    analyzer_model: str
    defender_model: str
    attack_objective: str
    strategy_name: str
    strategy_params: dict
    max_turns: int
    max_cost_usd: float
    created_at: datetime
    updated_at: datetime


class ExperimentListResponse(BaseModel):
    experiments: list[ExperimentResponse]
    total: int


# Sessions

class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    experiment_id: uuid.UUID
    status: str
    total_turns: int
    total_jailbreaks: int
    total_refused: int
    total_borderline: int
    total_blocked: int
    asr: float | None
    total_attacker_tokens: int
    total_target_tokens: int
    total_analyzer_tokens: int
    total_defender_tokens: int
    estimated_cost_usd: float
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime


class SessionListResponse(BaseModel):
    sessions: list[SessionResponse]
    total: int


# Turns

class TurnResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    session_id: uuid.UUID
    turn_number: int
    strategy_name: str
    strategy_params: dict
    attack_prompt: str
    target_response: str
    target_blocked: bool
    judge_verdict: str
    judge_confidence: float
    severity_score: float | None
    specificity_score: float | None
    coherence_score: float | None
    vulnerability_category: str | None
    attack_technique: str | None
    attacker_tokens: int
    target_tokens: int
    analyzer_tokens: int
    created_at: datetime


class TurnListResponse(BaseModel):
    turns: list[TurnResponse]
    total: int


# Strategies
class StrategyResponse(BaseModel):
    name: str
    display_name: str
    category: str
    description: str
    estimated_asr: str
    min_turns: int
    max_turns: int | None
    requires_white_box: bool
    supports_multi_turn: bool
    parameters: dict
    references: list[str]


class StrategyListResponse(BaseModel):
    strategies: list[StrategyResponse]
    total: int


# Targets / Models

class ModelListResponse(BaseModel):
    models: list[str]
    provider: str


class HealthResponse(BaseModel):
    provider: str
    healthy: bool


# WebSocket Messages
class WSMessage(BaseModel):
    """WebSocket message pushed to clients during live sessions."""

    type: str  # "turn_start", "turn_complete", "session_complete", "error"
    session_id: str
    turn_number: int | None = None
    data: dict = {}
