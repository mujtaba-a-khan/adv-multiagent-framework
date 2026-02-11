"""Session endpoints — create, list, get, and start sessions."""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException

from adversarial_framework.api.dependencies import (
    ExperimentRepoDep,
    OllamaProviderDep,
    SessionRepoDep,
)

if TYPE_CHECKING:
    from adversarial_framework.agents.target.providers.ollama import OllamaProvider
    from adversarial_framework.core.state import AttackTurn
    from adversarial_framework.db.models import Experiment

from adversarial_framework.api.schemas.requests import StartSessionRequest
from adversarial_framework.api.schemas.responses import (
    SessionListResponse,
    SessionResponse,
)

logger = structlog.get_logger(__name__)
router = APIRouter()

_SESSION_NOT_FOUND = "Session not found"
_EXPERIMENT_NOT_FOUND = "Experiment not found"


@dataclass(frozen=True)
class _ExperimentSnapshot:
    """Plain-data copy of an Experiment ORM object.

    Background tasks must not hold references to ORM objects because the
    DB session is closed after the request handler returns.  Accessing
    expired/lazy attributes outside the session raises a greenlet error.
    Snapshot all needed fields while the session is still open.

    Strategy and budget are NOT included here — they live on the Session
    since each session can use a different strategy.
    """

    id: uuid.UUID
    target_model: str
    target_provider: str
    target_system_prompt: str | None
    attacker_model: str
    analyzer_model: str
    defender_model: str
    attack_objective: str

    @classmethod
    def from_orm(cls, experiment: Experiment) -> _ExperimentSnapshot:
        return cls(
            id=experiment.id,
            target_model=experiment.target_model,
            target_provider=experiment.target_provider,
            target_system_prompt=experiment.target_system_prompt,
            attacker_model=experiment.attacker_model,
            analyzer_model=experiment.analyzer_model,
            defender_model=experiment.defender_model,
            attack_objective=experiment.attack_objective,
        )


@router.post(
    "/{experiment_id}/sessions",
    status_code=201,
    responses={404: {"description": "Experiment not found"}},
)
async def create_session(
    experiment_id: uuid.UUID,
    body: StartSessionRequest,
    experiment_repo: ExperimentRepoDep,
    session_repo: SessionRepoDep,
) -> SessionResponse:
    experiment = await experiment_repo.get(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail=_EXPERIMENT_NOT_FOUND)

    initial_defenses = (
        [d.model_dump() for d in body.initial_defenses] if body.initial_defenses else []
    )
    # Merge separate_reasoning into strategy_params for DB storage
    strategy_params = dict(body.strategy_params)
    strategy_params["separate_reasoning"] = body.separate_reasoning

    session = await session_repo.create(
        experiment_id=experiment_id,
        session_mode=body.session_mode,
        initial_defenses=initial_defenses,
        strategy_name=body.strategy_name,
        strategy_params=strategy_params,
        max_turns=body.max_turns,
        max_cost_usd=body.max_cost_usd,
    )
    return SessionResponse.model_validate(session)


@router.get("/{experiment_id}/sessions")
async def list_sessions(
    experiment_id: uuid.UUID,
    session_repo: SessionRepoDep,
    offset: int = 0,
    limit: int = 50,
) -> SessionListResponse:
    sessions, total = await session_repo.list_by_experiment(
        experiment_id, offset=offset, limit=limit
    )
    return SessionListResponse(
        sessions=[SessionResponse.model_validate(s) for s in sessions],
        total=total,
    )


@router.get(
    "/{experiment_id}/sessions/{session_id}",
    responses={404: {"description": "Session not found"}},
)
async def get_session(
    experiment_id: uuid.UUID,
    session_id: uuid.UUID,
    session_repo: SessionRepoDep,
) -> SessionResponse:
    session = await session_repo.get(session_id)
    if session is None or session.experiment_id != experiment_id:
        raise HTTPException(status_code=404, detail=_SESSION_NOT_FOUND)
    return SessionResponse.model_validate(session)


@router.delete(
    "/{experiment_id}/sessions/{session_id}",
    status_code=204,
    responses={
        404: {"description": "Session not found"},
        409: {"description": "Cannot delete a running session"},
    },
)
async def delete_session(
    experiment_id: uuid.UUID,
    session_id: uuid.UUID,
    session_repo: SessionRepoDep,
) -> None:
    """Delete a session and all its associated turns."""
    session = await session_repo.get(session_id)
    if session is None or session.experiment_id != experiment_id:
        raise HTTPException(status_code=404, detail=_SESSION_NOT_FOUND)

    if session.status == "running":
        raise HTTPException(status_code=409, detail="Cannot delete a running session")

    await session_repo.delete(session_id)


@router.post(
    "/{experiment_id}/sessions/{session_id}/start",
    responses={
        404: {"description": "Session or experiment not found"},
        409: {"description": "Session already started or completed"},
    },
)
async def start_session(
    experiment_id: uuid.UUID,
    session_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    experiment_repo: ExperimentRepoDep,
    session_repo: SessionRepoDep,
    provider: OllamaProviderDep,
) -> SessionResponse:
    """Start the adversarial attack loop in the background."""
    experiment = await experiment_repo.get(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail=_EXPERIMENT_NOT_FOUND)

    session = await session_repo.get(session_id)
    if session is None or session.experiment_id != experiment_id:
        raise HTTPException(status_code=404, detail=_SESSION_NOT_FOUND)

    if session.status != "pending":
        raise HTTPException(status_code=409, detail=f"Session is already '{session.status}'")

    # Snapshot experiment + session data BEFORE update_status() which
    # calls expire_all() and invalidates ORM objects in the identity map.
    experiment_snap = _ExperimentSnapshot.from_orm(experiment)
    session_mode = session.session_mode
    initial_defenses = session.initial_defenses or []
    strategy_name = session.strategy_name
    strategy_params = session.strategy_params or {}
    max_turns = session.max_turns
    max_cost_usd = session.max_cost_usd

    await session_repo.update_status(session_id, "running")

    # Commit the status + started_at update now so the background
    # task can freely write to the same session row without waiting
    # for this transaction to finish.
    await session_repo.db.commit()

    # Launch graph execution in background — pass plain snapshots,
    # not ORM objects, so the background task is decoupled from
    # the request's DB session.
    background_tasks.add_task(
        _run_graph_session,
        experiment=experiment_snap,
        session_id=session_id,
        provider=provider,
        session_mode=session_mode,
        initial_defenses=initial_defenses,
        strategy_name=strategy_name,
        strategy_params=strategy_params,
        max_turns=max_turns,
        max_cost_usd=max_cost_usd,
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


def _verdict_value(turn: AttackTurn) -> str:
    """Extract the string verdict from an AttackTurn."""
    v = turn.judge_verdict
    return v.value if hasattr(v, "value") else str(v)


def _turn_to_ws_payload(turn: AttackTurn, db_id: str, sid: str) -> dict[str, Any]:
    """Convert an AttackTurn + DB id into a WebSocket-friendly dict."""
    return {
        "id": db_id,
        "session_id": sid,
        "turn_number": turn.turn_number,
        "strategy_name": turn.strategy_name,
        "strategy_params": turn.strategy_params,
        "attack_prompt": turn.attack_prompt,
        "attacker_reasoning": getattr(turn, "attacker_reasoning", None),
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
    turn: AttackTurn,
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
            attacker_reasoning=getattr(turn, "attacker_reasoning", None),
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
    experiment: _ExperimentSnapshot,
    sid: str,
    *,
    session_mode: str,
    initial_defenses: list[dict[str, Any]] | None = None,
    strategy_name: str,
    strategy_params: dict[str, Any] | None = None,
    max_turns: int,
    max_cost_usd: float,
) -> dict[str, Any]:
    """Assemble the initial LangGraph state from experiment + session config."""
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
            "name": strategy_name,
            "params": strategy_params or {},
        },
        # Defense configuration
        "session_mode": session_mode,
        "initial_defense_config": initial_defenses or [],
        "defense_enabled": session_mode == "defense",
        # Execution control
        "status": SessionStatus.PENDING,
        "current_turn": 0,
        "max_turns": max_turns,
        "max_cost_usd": max_cost_usd,
        "selected_strategy": None,
        "strategy_params": None,
        "planning_notes": None,
        "current_attack_prompt": None,
        "attacker_reasoning": None,
        "target_response": None,
        "raw_target_response": None,
        "target_blocked": False,
        "judge_verdict": None,
        "judge_confidence": None,
        "judge_reason": None,
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
    experiment: _ExperimentSnapshot,
    session_id: uuid.UUID,
    provider: OllamaProvider,
    *,
    session_mode: str,
    initial_defenses: list[dict[str, Any]] | None = None,
    strategy_name: str,
    strategy_params: dict[str, Any] | None = None,
    max_turns: int,
    max_cost_usd: float,
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
            max_turns=max_turns,
            max_cost_usd=max_cost_usd,
            session_mode=session_mode,
            initial_defenses=initial_defenses,
            defense_model=experiment.defender_model,
        )
        compiled = graph.compile()
        initial_state = _build_initial_state(
            experiment,
            sid,
            session_mode=session_mode,
            initial_defenses=initial_defenses,
            strategy_name=strategy_name,
            strategy_params=strategy_params,
            max_turns=max_turns,
            max_cost_usd=max_cost_usd,
        )
        metrics = _SessionMetrics()

        async for event in compiled.astream(initial_state, stream_mode="updates"):  # type: ignore[arg-type]
            for node_name, update in event.items():
                await _handle_node_event(node_name, update, session_id, sid, metrics, broadcast)

    except Exception as exc:
        logger.error("graph_session_error", session_id=sid, error=str(exc))
        try:
            async with get_db_session() as db:
                await SessionRepository(db).update_status(session_id, "failed")
        except Exception:
            logger.error("failed_to_update_session_status", session_id=sid)
        with contextlib.suppress(Exception):
            await broadcast(
                sid,
                {
                    "type": "error",
                    "session_id": sid,
                    "turn_number": None,
                    "data": {"error": str(exc)},
                },
            )
    finally:
        await provider.close()


_BroadcastFn = Callable[[str, dict[str, Any]], Coroutine[Any, Any, None]]
_NodeHandler = Callable[
    [dict[str, Any], uuid.UUID, str, "_SessionMetrics", _BroadcastFn],
    Coroutine[Any, Any, None],
]


async def _handle_node_event(
    node_name: str,
    update: dict[str, Any],
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
    update: dict[str, Any],
    _sid: uuid.UUID,
    sid: str,
    metrics: _SessionMetrics,
    broadcast: _BroadcastFn,
) -> None:
    turn_num = metrics.total_turns + 1
    # For turn 1, _on_initialize already broadcast turn_start with the
    # attack_objective preview.  Skip the duplicate to avoid clobbering it.
    if turn_num > 1:
        await broadcast(
            sid,
            {
                "type": "turn_start",
                "session_id": sid,
                "turn_number": turn_num,
                "data": {},
            },
        )
    prompt = update.get("current_attack_prompt")
    if prompt:
        await broadcast(
            sid,
            {
                "type": "attack_generated",
                "session_id": sid,
                "turn_number": turn_num,
                "data": {
                    "attack_prompt": prompt,
                    "attacker_reasoning": update.get("attacker_reasoning"),
                    "strategy_name": update.get("selected_strategy", ""),
                    "is_baseline": turn_num == 1,
                },
            },
        )


async def _on_target(
    update: dict[str, Any],
    _sid: uuid.UUID,
    sid: str,
    metrics: _SessionMetrics,
    broadcast: _BroadcastFn,
) -> None:
    response = update.get("target_response")
    if response:
        await broadcast(
            sid,
            {
                "type": "target_responded",
                "session_id": sid,
                "turn_number": metrics.total_turns + 1,
                "data": {
                    "target_response": response,
                    "raw_target_response": update.get("raw_target_response"),
                    "target_blocked": update.get("target_blocked", False),
                },
            },
        )


async def _on_record_history(
    update: dict[str, Any],
    session_id: uuid.UUID,
    sid: str,
    metrics: _SessionMetrics,
    broadcast: _BroadcastFn,
) -> None:
    for turn in update.get("attack_history", []):
        _accumulate_metrics(turn, metrics)
        db_id = await _persist_turn_and_metrics(turn, session_id, metrics)
        await broadcast(
            sid,
            {
                "type": "turn_complete",
                "session_id": sid,
                "turn_number": turn.turn_number,
                "data": _turn_to_ws_payload(turn, db_id, sid),
            },
        )


def _accumulate_metrics(turn: AttackTurn, metrics: _SessionMetrics) -> None:
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
    _update: dict[str, Any],
    _session_id: uuid.UUID,
    sid: str,
    metrics: _SessionMetrics,
    broadcast: _BroadcastFn,
) -> None:
    """Broadcast turn_start immediately so the UI shows progress during cold-start.

    Includes the attack_objective so the frontend can display the baseline
    prompt before the attacker node completes (which may be delayed by
    Ollama model loading).
    """
    await broadcast(
        sid,
        {
            "type": "turn_start",
            "session_id": sid,
            "turn_number": 1,
            "data": {
                "attack_objective": _update.get("attack_objective", ""),
                "session_mode": _update.get("session_mode", "attack"),
            },
        },
    )


async def _on_defender(
    update: dict[str, Any],
    _sid: uuid.UUID,
    sid: str,
    metrics: _SessionMetrics,
    broadcast: _BroadcastFn,
) -> None:
    budget = update.get("token_budget")
    if budget and hasattr(budget, "total_defender_tokens"):
        metrics.defender_tokens += budget.total_defender_tokens
    await broadcast(
        sid,
        {
            "type": "defender_applied",
            "session_id": sid,
            "turn_number": None,
            "data": {"defender_tokens": metrics.defender_tokens},
        },
    )


async def _on_finalize(
    _update: dict[str, Any],
    session_id: uuid.UUID,
    sid: str,
    metrics: _SessionMetrics,
    broadcast: _BroadcastFn,
) -> None:
    from adversarial_framework.db.engine import get_session as get_db_session
    from adversarial_framework.db.repositories.sessions import SessionRepository

    async with get_db_session() as db:
        await SessionRepository(db).update_status(session_id, "completed")

    await broadcast(
        sid,
        {
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
        },
    )

    # Auto-detect baseline refusal for abliteration dataset
    await _check_baseline_for_dataset(session_id)


async def _check_baseline_for_dataset(
    session_id: uuid.UUID,
) -> None:
    """If the baseline turn was refused, suggest the objective for
    the abliteration dataset so the user can confirm it later."""
    from adversarial_framework.db.engine import get_session as get_db_session
    from adversarial_framework.db.repositories.abliteration_prompts import (
        AbliterationPromptRepository,
    )
    from adversarial_framework.db.repositories.sessions import (
        SessionRepository,
    )
    from adversarial_framework.db.repositories.turns import (
        TurnRepository,
    )

    try:
        async with get_db_session() as db:
            # Get baseline turn (turn_number=1)
            turn = await TurnRepository(db).get_by_turn_number(
                session_id,
                1,
            )
            if turn is None or turn.judge_verdict != "refused":
                return

            # Get the experiment objective via the session
            sess = await SessionRepository(db).get(session_id)
            if sess is None:
                return

            from adversarial_framework.db.repositories.experiments import (
                ExperimentRepository,
            )

            exp = await ExperimentRepository(db).get(sess.experiment_id)
            if exp is None:
                return

            objective = exp.attack_objective
            prompt_repo = AbliterationPromptRepository(db)

            # Deduplicate — skip if this text already exists
            if await prompt_repo.exists_by_text(objective):
                return

            await prompt_repo.create(
                text=objective,
                category="harmful",
                source="session",
                status="suggested",
                experiment_id=sess.experiment_id,
                session_id=session_id,
            )
    except Exception:
        logger.warning(
            "baseline_dataset_check_failed",
            session_id=str(session_id),
        )


_NODE_HANDLERS: dict[str, _NodeHandler] = {
    "initialize": _on_initialize,
    "attacker": _on_attacker,
    "target": _on_target,
    "record_history": _on_record_history,
    "defender": _on_defender,
    "finalize": _on_finalize,
}
