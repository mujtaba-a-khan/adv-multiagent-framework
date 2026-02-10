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
    session_mode: str
    initial_defenses: list[dict]
    strategy_name: str
    strategy_params: dict
    max_turns: int
    max_cost_usd: float
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
    attacker_reasoning: str | None
    target_response: str
    raw_target_response: str | None
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


# Defenses

class DefenseInfoResponse(BaseModel):
    """Metadata for a single registered defense mechanism."""

    name: str
    description: str
    defense_type: str


class DefenseListResponse(BaseModel):
    defenses: list[DefenseInfoResponse]
    total: int


# Comparisons

class SessionComparisonResponse(BaseModel):
    """Side-by-side comparison of an attack and defense session."""

    attack_session: SessionResponse
    defense_session: SessionResponse
    attack_turns: list[TurnResponse]
    defense_turns: list[TurnResponse]


# Fine-Tuning Jobs

class FineTuningJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    job_type: str
    source_model: str
    output_model_name: str
    config: dict
    status: str
    progress_pct: float
    current_step: str | None
    logs: list
    error_message: str | None
    peak_memory_gb: float | None
    total_duration_seconds: float | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime


class FineTuningJobListResponse(BaseModel):
    jobs: list[FineTuningJobResponse]
    total: int


# Abliteration Dataset


class AbliterationPromptResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    text: str
    category: str
    source: str
    status: str
    experiment_id: uuid.UUID | None
    session_id: uuid.UUID | None
    pair_id: uuid.UUID | None
    created_at: datetime


class AbliterationPromptListResponse(BaseModel):
    prompts: list[AbliterationPromptResponse]
    total: int


class AbliterationDatasetStatsResponse(BaseModel):
    harmful_count: int
    harmless_count: int
    total: int
    min_recommended: int = 16
    warning: str | None = None


class DatasetSuggestionResponse(BaseModel):
    suggestions: list[AbliterationPromptResponse]
    total: int


# Playground


class PlaygroundConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    target_model: str
    target_provider: str
    system_prompt: str | None
    analyzer_model: str
    active_defenses: list[dict]
    total_messages: int
    total_jailbreaks: int
    total_blocked: int
    total_target_tokens: int
    total_analyzer_tokens: int
    created_at: datetime
    updated_at: datetime


class PlaygroundConversationListResponse(BaseModel):
    conversations: list[PlaygroundConversationResponse]
    total: int


class PlaygroundMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    conversation_id: uuid.UUID
    message_number: int
    user_prompt: str
    target_response: str
    raw_target_response: str | None
    target_blocked: bool
    blocked_by_defense: str | None
    block_reason: str | None
    defenses_applied: list[dict] | None
    judge_verdict: str
    judge_confidence: float
    severity_score: float | None
    specificity_score: float | None
    vulnerability_category: str | None
    attack_technique: str | None
    target_tokens: int
    analyzer_tokens: int
    target_latency_ms: float
    analyzer_latency_ms: float
    created_at: datetime


class PlaygroundMessageListResponse(BaseModel):
    messages: list[PlaygroundMessageResponse]
    total: int


# WebSocket Messages
class WSMessage(BaseModel):
    """WebSocket message pushed to clients during live sessions."""

    type: str  # "turn_start", "turn_complete", "session_complete", "error"
    session_id: str
    turn_number: int | None = None
    data: dict = {}
