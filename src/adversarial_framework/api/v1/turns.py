"""Turn listing endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException

from adversarial_framework.api.dependencies import TurnRepoDep
from adversarial_framework.api.schemas.responses import TurnListResponse, TurnResponse

router = APIRouter()


@router.get("/{session_id}/turns")
async def list_turns(
    session_id: uuid.UUID,
    repo: TurnRepoDep,
    offset: int = 0,
    limit: int = 100,
) -> TurnListResponse:
    turns = await repo.list_by_session(session_id, offset=offset, limit=limit)
    return TurnListResponse(
        turns=[TurnResponse.model_validate(t) for t in turns],
        total=len(turns),
    )


@router.get("/{session_id}/turns/latest")
async def get_latest_turn(
    session_id: uuid.UUID,
    repo: TurnRepoDep,
) -> TurnResponse:
    turn = await repo.get_latest(session_id)
    if turn is None:
        raise HTTPException(status_code=404, detail="No turns found")
    return TurnResponse.model_validate(turn)
