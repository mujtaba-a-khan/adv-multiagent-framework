"""SQLAlchemy ORM models for the adversarial framework.

All tables use UUID primary keys and UTC timestamps.  The ``embedding``
columns use pgvector's ``Vector`` type for similarity search.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared base for all ORM models."""

    pass


# ── Experiments ──────────────────────────────────────────────────────────────


class Experiment(Base):
    """Top-level experiment configuration."""

    __tablename__ = "experiments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Target configuration
    target_model: Mapped[str] = mapped_column(String(128), nullable=False)
    target_provider: Mapped[str] = mapped_column(String(64), default="ollama")
    target_system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Agent model assignments
    attacker_model: Mapped[str] = mapped_column(String(128), nullable=False)
    analyzer_model: Mapped[str] = mapped_column(String(128), nullable=False)
    defender_model: Mapped[str] = mapped_column(String(128), nullable=False)

    # Attack configuration
    attack_objective: Mapped[str] = mapped_column(Text, nullable=False)
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_params: Mapped[dict] = mapped_column(JSON, default=dict)

    # Budget
    max_turns: Mapped[int] = mapped_column(Integer, default=20)
    max_cost_usd: Mapped[float] = mapped_column(Float, default=10.0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    sessions: Mapped[list[Session]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )


# ── Sessions ─────────────────────────────────────────────────────────────────


class Session(Base):
    """A single execution run of an experiment."""

    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False
    )

    # Status & mode
    status: Mapped[str] = mapped_column(String(32), default="pending")
    session_mode: Mapped[str] = mapped_column(String(32), default="attack", nullable=False)
    initial_defenses: Mapped[list] = mapped_column(JSON, default=list, nullable=False)

    # Session-level strategy and budget (set per-session at launch time)
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_params: Mapped[dict] = mapped_column(JSON, default=dict)
    max_turns: Mapped[int] = mapped_column(Integer, nullable=False)
    max_cost_usd: Mapped[float] = mapped_column(Float, nullable=False)

    # Aggregated metrics
    total_turns: Mapped[int] = mapped_column(Integer, default=0)
    total_jailbreaks: Mapped[int] = mapped_column(Integer, default=0)
    total_refused: Mapped[int] = mapped_column(Integer, default=0)
    total_borderline: Mapped[int] = mapped_column(Integer, default=0)
    total_blocked: Mapped[int] = mapped_column(Integer, default=0)
    asr: Mapped[float | None] = mapped_column(Float, nullable=True)  # attack success rate

    # Token usage
    total_attacker_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_target_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_analyzer_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_defender_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)

    # Timestamps
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    experiment: Mapped[Experiment] = relationship(back_populates="sessions")
    turns: Mapped[list[Turn]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="Turn.turn_number"
    )
    defense_actions: Mapped[list[DefenseActionRecord]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


# ── Turns ────────────────────────────────────────────────────────────────────


class Turn(Base):
    """Record of a single attack–response–analysis cycle."""

    __tablename__ = "turns"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    turn_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # Strategy
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_params: Mapped[dict] = mapped_column(JSON, default=dict)

    # Attack
    attack_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    attacker_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Target response
    target_response: Mapped[str] = mapped_column(Text, nullable=False)
    raw_target_response: Mapped[str | None] = mapped_column(Text, nullable=True)
    target_blocked: Mapped[bool] = mapped_column(Boolean, default=False)

    # Analysis
    judge_verdict: Mapped[str] = mapped_column(String(32), nullable=False)
    judge_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    severity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    specificity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    coherence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    vulnerability_category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    attack_technique: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Token usage
    attacker_tokens: Mapped[int] = mapped_column(Integer, default=0)
    target_tokens: Mapped[int] = mapped_column(Integer, default=0)
    analyzer_tokens: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    session: Mapped[Session] = relationship(back_populates="turns")


# ── Defense Actions ──────────────────────────────────────────────────────────


class DefenseActionRecord(Base):
    """Record of a defense generated or applied by the Defender agent."""

    __tablename__ = "defense_actions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )

    defense_type: Mapped[str] = mapped_column(String(64), nullable=False)
    defense_config: Mapped[dict] = mapped_column(JSON, default=dict)
    triggered_by_turn: Mapped[int] = mapped_column(Integer, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    block_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    false_positive_rate: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    session: Mapped[Session] = relationship(back_populates="defense_actions")


# Playground


class PlaygroundConversation(Base):
    """A user-driven playground chat session for manual LLM testing."""

    __tablename__ = "playground_conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)

    # Target configuration
    target_model: Mapped[str] = mapped_column(String(128), nullable=False)
    target_provider: Mapped[str] = mapped_column(String(64), default="ollama")
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Analyzer configuration
    analyzer_model: Mapped[str] = mapped_column(String(128), nullable=False)

    # Active defenses (JSON array of {name, params})
    active_defenses: Mapped[list] = mapped_column(
        JSON, default=list, nullable=False
    )

    # Aggregate stats (denormalized for fast sidebar display)
    total_messages: Mapped[int] = mapped_column(Integer, default=0)
    total_jailbreaks: Mapped[int] = mapped_column(Integer, default=0)
    total_blocked: Mapped[int] = mapped_column(Integer, default=0)
    total_target_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_analyzer_tokens: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    messages: Mapped[list[PlaygroundMessage]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="PlaygroundMessage.message_number",
    )


class PlaygroundMessage(Base):
    """A single exchange in a playground conversation."""

    __tablename__ = "playground_messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("playground_conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    message_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # User input
    user_prompt: Mapped[str] = mapped_column(Text, nullable=False)

    # Target response
    target_response: Mapped[str] = mapped_column(Text, nullable=False)
    raw_target_response: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    target_blocked: Mapped[bool] = mapped_column(Boolean, default=False)

    # Defense feedback
    blocked_by_defense: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    block_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Defense snapshot (which defenses were active when processed)
    defenses_applied: Mapped[list | None] = mapped_column(
        JSON, nullable=True, default=None
    )

    # Analysis
    judge_verdict: Mapped[str] = mapped_column(String(32), nullable=False)
    judge_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    severity_score: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    specificity_score: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    vulnerability_category: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    attack_technique: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )

    # Token usage
    target_tokens: Mapped[int] = mapped_column(Integer, default=0)
    analyzer_tokens: Mapped[int] = mapped_column(Integer, default=0)

    # Latency
    target_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    analyzer_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    conversation: Mapped[PlaygroundConversation] = relationship(
        back_populates="messages"
    )


# ── Vulnerability Catalog ────────────────────────────────────────────────────


class VulnerabilityCatalog(Base):
    """Reference table for OWASP LLM Top 10 / NIST AI RMF categories."""

    __tablename__ = "vulnerability_catalog"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)  # e.g. "LLM01"
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    framework: Mapped[str] = mapped_column(String(32), nullable=False)  # "owasp" | "nist"
    severity_default: Mapped[float] = mapped_column(Float, default=5.0)


# Fine-Tuning Jobs

class FineTuningJob(Base):
    """A model fine-tuning or abliteration job."""

    __tablename__ = "finetuning_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # Job identity
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    job_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # "pull_abliterated" | "abliterate" | "sft"

    # Source and output
    source_model: Mapped[str] = mapped_column(String(255), nullable=False)
    output_model_name: Mapped[str] = mapped_column(
        String(255), nullable=False
    )

    # Job configuration (type-specific params stored as JSON)
    config: Mapped[dict] = mapped_column(JSON, default=dict)

    # Status tracking
    status: Mapped[str] = mapped_column(
        String(32), default="pending"
    )  # pending | running | completed | failed | cancelled
    progress_pct: Mapped[float] = mapped_column(Float, default=0.0)
    current_step: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )

    # Logs (append-only JSON array of log entries)
    logs: Mapped[list] = mapped_column(JSON, default=list)

    # Error info
    error_message: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )

    # Resource tracking
    peak_memory_gb: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    total_duration_seconds: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )

    # Timestamps
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ── Session Reports ──────────────────────────────────────────────────────────


class SessionReport(Base):
    """Generated report summary for a completed session."""

    __tablename__ = "session_reports"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, unique=True
    )

    summary: Mapped[str] = mapped_column(Text, nullable=False)
    findings: Mapped[dict] = mapped_column(JSON, default=dict)
    recommendations: Mapped[dict] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# Abliteration Dataset

class AbliterationPrompt(Base):
    """A prompt in the abliteration dataset (harmful or harmless counterpart)."""

    __tablename__ = "abliteration_prompts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(
        String(16), nullable=False
    )  # "harmful" | "harmless"
    source: Mapped[str] = mapped_column(
        String(32), default="manual"
    )  # "manual" | "upload" | "session"
    status: Mapped[str] = mapped_column(
        String(16), default="active", server_default="active"
    )  # "active" | "suggested"

    experiment_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiments.id", ondelete="SET NULL"),
        nullable=True,
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="SET NULL"),
        nullable=True,
    )
    pair_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("abliteration_prompts.id", ondelete="SET NULL"),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
