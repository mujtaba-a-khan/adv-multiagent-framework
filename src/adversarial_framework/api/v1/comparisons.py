"""Session comparison endpoints for attack vs defense analysis."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException

from adversarial_framework.api.dependencies import (
    get_session_repo,
    get_turn_repo,
)
from adversarial_framework.api.schemas.responses import (
    SessionComparisonResponse,
    SessionResponse,
    TurnResponse,
)
from adversarial_framework.db.repositories.sessions import SessionRepository
from adversarial_framework.db.repositories.turns import TurnRepository

router = APIRouter()


@router.get(
    "/{experiment_id}/compare",
    response_model=SessionComparisonResponse,
)
async def compare_sessions(
    experiment_id: uuid.UUID,
    attack_session_id: uuid.UUID,
    defense_session_id: uuid.UUID,
    session_repo: SessionRepository = Depends(get_session_repo),
    turn_repo: TurnRepository = Depends(get_turn_repo),
) -> SessionComparisonResponse:
    """Compare an attack session with a defense session side-by-side."""
    attack_session = await session_repo.get(attack_session_id)
    if attack_session is None or attack_session.experiment_id != experiment_id:
        raise HTTPException(status_code=404, detail="Attack session not found")

    defense_session = await session_repo.get(defense_session_id)
    if defense_session is None or defense_session.experiment_id != experiment_id:
        raise HTTPException(status_code=404, detail="Defense session not found")

    if attack_session.session_mode != "attack":
        raise HTTPException(
            status_code=400,
            detail="First session must be an attack session",
        )
    if defense_session.session_mode != "defense":
        raise HTTPException(
            status_code=400,
            detail="Second session must be a defense session",
        )

    attack_turns = await turn_repo.list_by_session(attack_session_id)
    defense_turns = await turn_repo.list_by_session(defense_session_id)

    return SessionComparisonResponse(
        attack_session=SessionResponse.model_validate(attack_session),
        defense_session=SessionResponse.model_validate(defense_session),
        attack_turns=[TurnResponse.model_validate(t) for t in attack_turns],
        defense_turns=[TurnResponse.model_validate(t) for t in defense_turns],
    )
