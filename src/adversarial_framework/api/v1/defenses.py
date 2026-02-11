"""Defense catalog endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from adversarial_framework.api.schemas.responses import (
    DefenseInfoResponse,
    DefenseListResponse,
)
from adversarial_framework.defenses.registry import DefenseRegistry

router = APIRouter()


@router.get("")
async def list_defenses() -> DefenseListResponse:
    """List all registered defense mechanisms."""
    DefenseRegistry.discover()
    defenses = [
        DefenseInfoResponse(
            name=name,
            description=(DefenseRegistry.get(name).__doc__ or "").strip(),
            defense_type=name,
        )
        for name in DefenseRegistry.list_names()
    ]
    return DefenseListResponse(defenses=defenses, total=len(defenses))
