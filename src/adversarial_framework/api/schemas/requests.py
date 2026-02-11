"""Pydantic request schemas for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from adversarial_framework.config.settings import get_settings

_settings = get_settings()


class CreateExperimentRequest(BaseModel):
    """Request body for creating a new experiment."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None

    # Target
    target_model: str = Field(..., min_length=1, max_length=128)
    target_provider: str = Field(default="ollama", max_length=64)
    target_system_prompt: str | None = None

    # Agent models
    attacker_model: str = Field(default=_settings.attacker_model, max_length=128)
    analyzer_model: str = Field(default=_settings.analyzer_model, max_length=128)
    defender_model: str = Field(default=_settings.defender_model, max_length=128)

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


class DefenseSelectionItem(BaseModel):
    """A single defense selection with optional parameter overrides."""

    name: str = Field(..., min_length=1, max_length=64)
    params: dict = Field(default_factory=dict)


class StartSessionRequest(BaseModel):
    """Request body for creating a new session with mode, strategy, and budget."""

    session_mode: str = Field(default="attack", pattern=r"^(attack|defense)$")
    initial_defenses: list[DefenseSelectionItem] | None = Field(default=None)

    # Strategy and budget (required — user selects at launch time)
    strategy_name: str = Field(..., min_length=1, max_length=64)
    strategy_params: dict = Field(default_factory=dict)
    max_turns: int = Field(..., ge=1, le=100)
    max_cost_usd: float = Field(..., ge=0.0, le=500.0)

    # Reasoning separation toggle
    separate_reasoning: bool = Field(
        default=True,
        description=(
            "When True, LLM-based strategies use a two-call pattern "
            "(think + attack) to separate reasoning from the clean "
            "prompt. When False, uses the original single-call pattern."
        ),
    )


class CreateFineTuningJobRequest(BaseModel):
    """Request body for creating a fine-tuning or abliteration job."""

    name: str = Field(..., min_length=1, max_length=255)
    job_type: str = Field(..., pattern=r"^(pull_abliterated|abliterate|sft)$")
    source_model: str = Field(..., min_length=1, max_length=255)
    output_model_name: str = Field(..., min_length=1, max_length=255)
    config: dict = Field(default_factory=dict)


# Abliteration Dataset


class CreateAbliterationPromptRequest(BaseModel):
    """Request body for adding a prompt to the abliteration dataset."""

    text: str = Field(..., min_length=1)
    category: str = Field(..., pattern=r"^(harmful|harmless)$")
    source: str = Field(default="manual", pattern=r"^(manual|upload|session)$")
    experiment_id: str | None = None
    session_id: str | None = None
    auto_generate_counterpart: bool = False


class UpdateAbliterationPromptRequest(BaseModel):
    """Request body for updating an abliteration prompt."""

    text: str | None = Field(default=None, min_length=1)
    category: str | None = Field(default=None, pattern=r"^(harmful|harmless)$")


class GenerateHarmlessRequest(BaseModel):
    """Request to generate a harmless counterpart for a harmful prompt."""

    harmful_prompt: str = Field(..., min_length=1)
    pair_id: str | None = None
    experiment_id: str | None = None
    session_id: str | None = None


class ModelLoadRequest(BaseModel):
    """Request to load or unload an Ollama model."""

    model: str = Field(..., min_length=1, max_length=255)


# Playground


class CreatePlaygroundConversationRequest(BaseModel):
    """Request body for creating a new playground conversation."""

    title: str = Field(..., min_length=1, max_length=255)
    target_model: str = Field(..., min_length=1, max_length=128)
    target_provider: str = Field(default="ollama", max_length=64)
    system_prompt: str | None = None
    analyzer_model: str = Field(default=_settings.analyzer_model, max_length=128)
    active_defenses: list[DefenseSelectionItem] | None = None


class UpdatePlaygroundConversationRequest(BaseModel):
    """Request body for updating a playground conversation."""

    title: str | None = Field(default=None, min_length=1, max_length=255)
    system_prompt: str | None = None
    active_defenses: list[DefenseSelectionItem] | None = None


class SendPlaygroundMessageRequest(BaseModel):
    """Request body for sending a message in a playground conversation."""

    prompt: str = Field(..., min_length=1)
