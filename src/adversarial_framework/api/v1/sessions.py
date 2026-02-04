"""Session endpoints — create, list, get, and start sessions."""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from adversarial_framework.agents.target.providers.ollama import OllamaProvider
from adversarial_framework.api.dependencies import (
    get_experiment_repo,
    get_ollama_provider,
    get_session_repo,
)
from adversarial_framework.api.schemas.requests import StartSessionRequest
from adversarial_framework.api.schemas.responses import (
    SessionListResponse,
    SessionResponse,
)
from adversarial_framework.db.repositories.experiments import ExperimentRepository
from adversarial_framework.db.repositories.sessions import SessionRepository

logger = structlog.get_logger(__name__)
router = APIRouter()


@dataclass(frozen=True)
class _ExperimentSnapshot:
    """Plain-data copy of an Experiment ORM object.

    Background tasks must not hold references to ORM objects because the
    DB session is closed after the request handler returns.  Accessing
    expired/lazy attributes outside the session raises a greenlet error.
    Snapshot all needed fields while the session is still open.
    """

    id: uuid.UUID
    target_model: str
    target_provider: str
    target_system_prompt: str | None
    attacker_model: str
    analyzer_model: str
    defender_model: str
    attack_objective: str
    strategy_name: str
    strategy_params: dict[str, Any] | None
    max_turns: int
    max_cost_usd: float

    @classmethod
    def from_orm(cls, experiment: object) -> _ExperimentSnapshot:
        return cls(
            id=experiment.id,
            target_model=experiment.target_model,
            target_provider=experiment.target_provider,
            target_system_prompt=experiment.target_system_prompt,
            attacker_model=experiment.attacker_model,
            analyzer_model=experiment.analyzer_model,
            defender_model=experiment.defender_model,
            attack_objective=experiment.attack_objective,
            strategy_name=experiment.strategy_name,
            strategy_params=experiment.strategy_params,
            max_turns=experiment.max_turns,
            max_cost_usd=experiment.max_cost_usd,
        )


@router.post(
    "/{experiment_id}/sessions",
    response_model=SessionResponse,
    status_code=201,
)
async def create_session(
    experiment_id: uuid.UUID,
    body: StartSessionRequest | None = None,
    experiment_repo: ExperimentRepository = Depends(get_experiment_repo),
    session_repo: SessionRepository = Depends(get_session_repo),
) -> SessionResponse:
    experiment = await experiment_repo.get(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    session_mode = body.session_mode if body else "attack"
    initial_defenses = (
        [d.model_dump() for d in body.initial_defenses]
        if body and body.initial_defenses
        else []
    )
    session = await session_repo.create(
        experiment_id=experiment_id,
        session_mode=session_mode,
        initial_defenses=initial_defenses,
    )
    return SessionResponse.model_validate(session)


@router.get(
    "/{experiment_id}/sessions",
    response_model=SessionListResponse,
)
async def list_sessions(
    experiment_id: uuid.UUID,
    offset: int = 0,
    limit: int = 50,
    session_repo: SessionRepository = Depends(get_session_repo),
) -> SessionListResponse:
    sessions = await session_repo.list_by_experiment(
        experiment_id, offset=offset, limit=limit
    )
    return SessionListResponse(
        sessions=[SessionResponse.model_validate(s) for s in sessions],
        total=len(sessions),
    )


@router.get(
    "/{experiment_id}/sessions/{session_id}",
    response_model=SessionResponse,
)
async def get_session(
    experiment_id: uuid.UUID,
    session_id: uuid.UUID,
    session_repo: SessionRepository = Depends(get_session_repo),
) -> SessionResponse:
    session = await session_repo.get(session_id)
    if session is None or session.experiment_id != experiment_id:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse.model_validate(session)


@router.post(
    "/{experiment_id}/sessions/{session_id}/start",
    response_model=SessionResponse,
)
async def start_session(
    experiment_id: uuid.UUID,
    session_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    experiment_repo: ExperimentRepository = Depends(get_experiment_repo),
    session_repo: SessionRepository = Depends(get_session_repo),
    provider: OllamaProvider = Depends(get_ollama_provider),
) -> SessionResponse:
    """Start the adversarial attack loop in the background."""
    experiment = await experiment_repo.get(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    session = await session_repo.get(session_id)
    if session is None or session.experiment_id != experiment_id:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status != "pending":
        raise HTTPException(
            status_code=409, detail=f"Session is already '{session.status}'"
        )

    # Snapshot experiment data BEFORE update_status() which calls
    # expire_all() and invalidates ORM objects in the identity map.
    experiment_snap = _ExperimentSnapshot.from_orm(experiment)
    session_mode = session.session_mode
    initial_defenses = session.initial_defenses or []

    await session_repo.update_status(session_id, "running")

    # Launch graph execution in background — pass the plain snapshot,
    # not the ORM object, so the background task is decoupled from
    # the request's DB session.
    background_tasks.add_task(
        _run_graph_session,
        experiment=experiment_snap,
        session_id=session_id,
        provider=provider,
        session_mode=session_mode,
        initial_defenses=initial_defenses,
    )

    # Re-fetch to get updated status
    session = await session_repo.get(session_id)
    return SessionResponse.model_validate(session)


# ── Streaming helpers ────────────────────────────────────────────────────────


@dataclass
class _SessionMetrics:
    """Mutable accumulator for session-level metrics during streaming."""

    total_turns: int = 0
    total_jailbreaks: int = 0
    total_borderline: int = 0
    total_refused: int = 0
    total_blocked: int = 0
    attacker_tokens: int = 0
    target_tokens: int = 0
    analyzer_tokens: int = 0
    defender_tokens: int = 0


def _verdict_value(turn: object) -> str:
    """Extract the string verdict from an AttackTurn."""
    v = turn.judge_verdict
    return v.value if hasattr(v, "value") else str(v)


def _turn_to_ws_payload(turn: object, db_id: str, sid: str) -> dict:
    """Convert an AttackTurn + DB id into a WebSocket-friendly dict."""
    return {
        "id": db_id,
        "session_id": sid,
        "turn_number": turn.turn_number,
        "strategy_name": turn.strategy_name,
        "strategy_params": turn.strategy_params,
        "attack_prompt": turn.attack_prompt,
        "target_response": turn.target_response,
        "raw_target_response": getattr(turn, "raw_target_response", None),
        "target_blocked": turn.target_blocked,
        "judge_verdict": _verdict_value(turn),
        "judge_confidence": turn.confidence,
        "severity_score": turn.severity,
        "specificity_score": turn.specificity,
        "coherence_score": None,
        "vulnerability_category": turn.vulnerability_category,
        "attack_technique": turn.attack_technique,
        "attacker_tokens": turn.attacker_tokens,
        "target_tokens": turn.target_tokens,
        "analyzer_tokens": turn.analyzer_tokens,
        "created_at": turn.timestamp.isoformat(),
    }


async def _persist_turn_and_metrics(
    turn: object,
    session_id: uuid.UUID,
    metrics: _SessionMetrics,
) -> str:
    """Save a turn to the DB, update session metrics, return the DB turn id."""
    from adversarial_framework.db.engine import get_session as get_db_session
    from adversarial_framework.db.repositories.sessions import SessionRepository
    from adversarial_framework.db.repositories.turns import TurnRepository

    verdict_val = _verdict_value(turn)

    async with get_db_session() as db:
        db_turn = await TurnRepository(db).create(
            session_id=session_id,
            turn_number=turn.turn_number,
            strategy_name=turn.strategy_name,
            strategy_params=turn.strategy_params,
            attack_prompt=turn.attack_prompt,
            target_response=turn.target_response,
            raw_target_response=getattr(turn, "raw_target_response", None),
            target_blocked=turn.target_blocked,
            judge_verdict=verdict_val,
            judge_confidence=turn.confidence,
            severity_score=turn.severity,
            specificity_score=turn.specificity,
            vulnerability_category=turn.vulnerability_category,
            attack_technique=turn.attack_technique,
            attacker_tokens=turn.attacker_tokens,
            target_tokens=turn.target_tokens,
            analyzer_tokens=turn.analyzer_tokens,
        )

        await SessionRepository(db).update_metrics(
            session_id=session_id,
            total_turns=metrics.total_turns,
            total_jailbreaks=metrics.total_jailbreaks,
            total_borderline=metrics.total_borderline,
            total_refused=metrics.total_refused,
            total_blocked=metrics.total_blocked,
            total_attacker_tokens=metrics.attacker_tokens,
            total_target_tokens=metrics.target_tokens,
            total_analyzer_tokens=metrics.analyzer_tokens,
            total_defender_tokens=metrics.defender_tokens,
            estimated_cost_usd=0.0,
        )

    return str(db_turn.id)


def _build_initial_state(
    experiment: object,
    sid: str,
    session_mode: str = "attack",
    initial_defenses: list[dict[str, Any]] | None = None,
) -> dict:
    """Assemble the initial LangGraph state from an experiment record."""
    from adversarial_framework.core.constants import SessionStatus
    from adversarial_framework.core.state import TokenBudget

    return {
        "experiment_id": str(experiment.id),
        "session_id": sid,
        "target_model": experiment.target_model,
        "target_provider": experiment.target_provider,
        "target_system_prompt": experiment.target_system_prompt,
        "attacker_model": experiment.attacker_model,
        "analyzer_model": experiment.analyzer_model,
        "defender_model": experiment.defender_model,
        "attack_objective": experiment.attack_objective,
        "strategy_config": {
            "name": experiment.strategy_name,
            "params": experiment.strategy_params or {},
        },
        # Defense configuration
        "session_mode": session_mode,
        "initial_defense_config": initial_defenses or [],
        "defense_enabled": session_mode == "defense",
        # Execution control
        "status": SessionStatus.PENDING,
        "current_turn": 0,
        "max_turns": experiment.max_turns,
        "max_cost_usd": experiment.max_cost_usd,
        "selected_strategy": None,
        "strategy_params": None,
        "planning_notes": None,
        "current_attack_prompt": None,
        "target_response": None,
        "raw_target_response": None,
        "target_blocked": False,
        "judge_verdict": None,
        "judge_confidence": None,
        "severity_score": None,
        "specificity_score": None,
        "coherence_score": None,
        "vulnerability_category": None,
        "attack_technique": None,
        "attack_history": [],
        "defense_actions": [],
        "token_budget": TokenBudget(),
        "total_jailbreaks": 0,
        "total_borderline": 0,
        "total_blocked": 0,
        "total_refused": 0,
        "conversation_history": [],
        "human_feedback": None,
        "awaiting_human": False,
        "last_error": None,
        "error_count": 0,
    }


# ── Background graph runner ──────────────────────────────────────────────────


async def _run_graph_session(
    experiment: object,
    session_id: uuid.UUID,
    provider: OllamaProvider,
    session_mode: str = "attack",
    initial_defenses: list[dict[str, Any]] | None = None,
) -> None:
    """Execute the LangGraph adversarial loop with real-time streaming.

    Uses ``astream(stream_mode="updates")`` to process each node's output
    as it completes, persisting turns to the database and broadcasting
    WebSocket messages incrementally.
    """
    from adversarial_framework.api.v1.ws import broadcast
    from adversarial_framework.core.graph import build_graph
    from adversarial_framework.db.engine import get_session as get_db_session
    from adversarial_framework.db.repositories.sessions import SessionRepository

    sid = str(session_id)

    try:
        graph = build_graph(
            provider=provider,
            attacker_config={"model": experiment.attacker_model},
            analyzer_config={"model": experiment.analyzer_model},
            target_system_prompt=experiment.target_system_prompt,
            max_turns=experiment.max_turns,
            max_cost_usd=experiment.max_cost_usd,
            session_mode=session_mode,
            initial_defenses=initial_defenses,
            defense_model=experiment.defender_model,
        )
        compiled = graph.compile()
        initial_state = _build_initial_state(
            experiment, sid,
            session_mode=session_mode,
            initial_defenses=initial_defenses,
        )
        metrics = _SessionMetrics()

        async for event in compiled.astream(
            initial_state, stream_mode="updates"
        ):
            for node_name, update in event.items():
                await _handle_node_event(
                    node_name, update, session_id, sid, metrics, broadcast
                )

    except Exception as exc:
        logger.error("graph_session_error", session_id=sid, error=str(exc))
        try:
            async with get_db_session() as db:
                await SessionRepository(db).update_status(session_id, "failed")
        except Exception:
            logger.error("failed_to_update_session_status", session_id=sid)
        with contextlib.suppress(Exception):
            await broadcast(sid, {
                "type": "error",
                "session_id": sid,
                "turn_number": None,
                "data": {"error": str(exc)},
            })
    finally:
        await provider.close()


_BroadcastFn = Callable[[str, dict], Coroutine[Any, Any, None]]
_NodeHandler = Callable[
    [dict, uuid.UUID, str, "_SessionMetrics", _BroadcastFn],
    Coroutine[Any, Any, None],
]


async def _handle_node_event(
    node_name: str,
    update: dict,
    session_id: uuid.UUID,
    sid: str,
    metrics: _SessionMetrics,
    broadcast: _BroadcastFn,
) -> None:
    """Dispatch a single node's output to the appropriate handler."""
    handler = _NODE_HANDLERS.get(node_name)
    if handler is not None:
        await handler(update, session_id, sid, metrics, broadcast)


async def _on_attacker(
    update: dict, _sid: uuid.UUID, sid: str,
    metrics: _SessionMetrics, broadcast: _BroadcastFn,
) -> None:
    turn_num = metrics.total_turns + 1
    # For turn 1, _on_initialize already broadcast turn_start with the
    # attack_objective preview.  Skip the duplicate to avoid clobbering it.
    if turn_num > 1:
        await broadcast(sid, {
            "type": "turn_start",
            "session_id": sid,
            "turn_number": turn_num,
            "data": {},
        })
    prompt = update.get("current_attack_prompt")
    if prompt:
        await broadcast(sid, {
            "type": "attack_generated",
            "session_id": sid,
            "turn_number": turn_num,
            "data": {
                "attack_prompt": prompt,
                "strategy_name": update.get("selected_strategy", ""),
                "is_baseline": turn_num == 1,
            },
        })


async def _on_target(
    update: dict, _sid: uuid.UUID, sid: str,
    metrics: _SessionMetrics, broadcast: _BroadcastFn,
) -> None:
    response = update.get("target_response")
    if response:
        await broadcast(sid, {
            "type": "target_responded",
            "session_id": sid,
            "turn_number": metrics.total_turns + 1,
            "data": {
                "target_response": response,
                "raw_target_response": update.get("raw_target_response"),
                "target_blocked": update.get("target_blocked", False),
            },
        })


async def _on_record_history(
    update: dict, session_id: uuid.UUID, sid: str,
    metrics: _SessionMetrics, broadcast: _BroadcastFn,
) -> None:
    for turn in update.get("attack_history", []):
        _accumulate_metrics(turn, metrics)
        db_id = await _persist_turn_and_metrics(turn, session_id, metrics)
        await broadcast(sid, {
            "type": "turn_complete",
            "session_id": sid,
            "turn_number": turn.turn_number,
            "data": _turn_to_ws_payload(turn, db_id, sid),
        })


def _accumulate_metrics(turn: object, metrics: _SessionMetrics) -> None:
    """Update running metric counters from a completed turn."""
    verdict_val = _verdict_value(turn)
    metrics.total_turns += 1
    if verdict_val == "jailbreak":
        metrics.total_jailbreaks += 1
    elif verdict_val == "borderline":
        metrics.total_borderline += 1
    elif verdict_val == "refused":
        metrics.total_refused += 1
    if turn.target_blocked:
        metrics.total_blocked += 1
    metrics.attacker_tokens += turn.attacker_tokens
    metrics.target_tokens += turn.target_tokens
    metrics.analyzer_tokens += turn.analyzer_tokens


async def _on_initialize(
    _update: dict, _session_id: uuid.UUID, sid: str,
    metrics: _SessionMetrics, broadcast: _BroadcastFn,
) -> None:
    """Broadcast turn_start immediately so the UI shows progress during cold-start.

    Includes the attack_objective so the frontend can display the baseline
    prompt before the attacker node completes (which may be delayed by
    Ollama model loading).
    """
    await broadcast(sid, {
        "type": "turn_start",
        "session_id": sid,
        "turn_number": 1,
        "data": {
            "attack_objective": _update.get("attack_objective", ""),
            "session_mode": _update.get("session_mode", "attack"),
        },
    })


async def _on_defender(
    update: dict, _sid: uuid.UUID, _s: str,
    metrics: _SessionMetrics, _broadcast: _BroadcastFn,
) -> None:
    budget = update.get("token_budget")
    if budget and hasattr(budget, "total_defender_tokens"):
        metrics.defender_tokens += budget.total_defender_tokens


async def _on_finalize(
    _update: dict, session_id: uuid.UUID, sid: str,
    metrics: _SessionMetrics, broadcast: _BroadcastFn,
) -> None:
    from adversarial_framework.db.engine import get_session as get_db_session
    from adversarial_framework.db.repositories.sessions import SessionRepository

    async with get_db_session() as db:
        await SessionRepository(db).update_status(session_id, "completed")

    await broadcast(sid, {
        "type": "session_complete",
        "session_id": sid,
        "turn_number": None,
        "data": {
            "total_turns": metrics.total_turns,
            "total_jailbreaks": metrics.total_jailbreaks,
            "total_borderline": metrics.total_borderline,
            "total_refused": metrics.total_refused,
            "total_blocked": metrics.total_blocked,
        },
    })


_NODE_HANDLERS: dict[str, _NodeHandler] = {
    "initialize": _on_initialize,
    "attacker": _on_attacker,
    "target": _on_target,
    "record_history": _on_record_history,
    "defender": _on_defender,
    "finalize": _on_finalize,
}
