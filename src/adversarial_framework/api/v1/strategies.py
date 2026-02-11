"""Strategy catalog endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from adversarial_framework.api.schemas.responses import (
    StrategyListResponse,
    StrategyResponse,
)
from adversarial_framework.strategies.registry import StrategyRegistry

router = APIRouter()


@router.get("")
async def list_strategies() -> StrategyListResponse:
    """List all registered attack strategies."""
    StrategyRegistry.discover()
    all_meta = StrategyRegistry.list_all()
    strategies = [
        StrategyResponse(
            name=m.name,
            display_name=m.display_name,
            category=m.category.value if hasattr(m.category, "value") else str(m.category),
            description=m.description,
            estimated_asr=m.estimated_asr,
            min_turns=m.min_turns,
            max_turns=m.max_turns,
            requires_white_box=m.requires_white_box,
            supports_multi_turn=m.supports_multi_turn,
            parameters=m.parameters,
            references=m.references,
        )
        for m in all_meta
    ]
    return StrategyListResponse(strategies=strategies, total=len(strategies))


@router.get("/{strategy_name}")
async def get_strategy(strategy_name: str) -> StrategyResponse:
    """Get details of a specific attack strategy."""
    StrategyRegistry.discover()
    try:
        cls = StrategyRegistry.get(strategy_name)
    except KeyError as err:
        raise HTTPException(
            status_code=404,
            detail=f"Strategy '{strategy_name}' not found. "
            f"Available: {StrategyRegistry.list_names()}",
        ) from err
    m = cls.metadata
    return StrategyResponse(
        name=m.name,
        display_name=m.display_name,
        category=m.category.value if hasattr(m.category, "value") else str(m.category),
        description=m.description,
        estimated_asr=m.estimated_asr,
        min_turns=m.min_turns,
        max_turns=m.max_turns,
        requires_white_box=m.requires_white_box,
        supports_multi_turn=m.supports_multi_turn,
        parameters=m.parameters,
        references=m.references,
    )
