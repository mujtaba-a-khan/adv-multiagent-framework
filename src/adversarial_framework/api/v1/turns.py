"""Turn listing endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException

from adversarial_framework.api.dependencies import get_turn_repo
from adversarial_framework.api.schemas.responses import TurnListResponse, TurnResponse
from adversarial_framework.db.repositories.turns import TurnRepository

router = APIRouter()


@router.get("/{session_id}/turns", response_model=TurnListResponse)
async def list_turns(
    session_id: uuid.UUID,
    offset: int = 0,
    limit: int = 100,
    repo: TurnRepository = Depends(get_turn_repo),
) -> TurnListResponse:
    turns = await repo.list_by_session(session_id, offset=offset, limit=limit)
    return TurnListResponse(
        turns=[TurnResponse.model_validate(t) for t in turns],
        total=len(turns),
    )


@router.get("/{session_id}/turns/latest", response_model=TurnResponse)
async def get_latest_turn(
    session_id: uuid.UUID,
    repo: TurnRepository = Depends(get_turn_repo),
) -> TurnResponse:
    turn = await repo.get_latest(session_id)
    if turn is None:
        raise HTTPException(status_code=404, detail="No turns found")
    return TurnResponse.model_validate(turn)
