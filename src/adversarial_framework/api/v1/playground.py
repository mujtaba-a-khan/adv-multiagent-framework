"""Playground API for freestyle manual LLM testing with analysis and defenses."""

from __future__ import annotations

import re
import time
import uuid
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException

from adversarial_framework.agents.analyzer.classifier import (
    run_classifier,
)
from adversarial_framework.agents.analyzer.judge import run_judge
from adversarial_framework.agents.analyzer.scorer import run_scorer
from adversarial_framework.agents.target.interface import TargetInterface
from adversarial_framework.agents.target.providers.base import BaseProvider
from adversarial_framework.agents.target.providers.ollama import (
    OllamaProvider,
)
from adversarial_framework.api.dependencies import (
    get_ollama_provider,
    get_playground_repo,
)
from adversarial_framework.api.schemas.requests import (
    CreatePlaygroundConversationRequest,
    SendPlaygroundMessageRequest,
    UpdatePlaygroundConversationRequest,
)
from adversarial_framework.api.schemas.responses import (
    PlaygroundConversationListResponse,
    PlaygroundConversationResponse,
    PlaygroundMessageListResponse,
    PlaygroundMessageResponse,
)
from adversarial_framework.api.v1.ws import broadcast_playground
from adversarial_framework.core.constants import JudgeVerdict
from adversarial_framework.db.repositories.playground import (
    PlaygroundRepository,
)
from adversarial_framework.defenses.base import BaseDefense
from adversarial_framework.defenses.registry import DefenseRegistry

logger = structlog.get_logger(__name__)
router = APIRouter()

_PROVIDER_DEFENSES = {"llm_judge", "two_pass"}

# Conversation CRUD 

@router.post(
    "/conversations",
    response_model=PlaygroundConversationResponse,
    status_code=201,
)
async def create_conversation(
    body: CreatePlaygroundConversationRequest,
    repo: PlaygroundRepository = Depends(get_playground_repo),
) -> PlaygroundConversationResponse:
    """Create a new playground conversation."""
    defenses_data = (
        [d.model_dump() for d in body.active_defenses]
        if body.active_defenses
        else []
    )
    conversation = await repo.create_conversation(
        title=body.title,
        target_model=body.target_model,
        target_provider=body.target_provider,
        system_prompt=body.system_prompt,
        analyzer_model=body.analyzer_model,
        active_defenses=defenses_data,
    )
    return PlaygroundConversationResponse.model_validate(
        conversation
    )


@router.get(
    "/conversations",
    response_model=PlaygroundConversationListResponse,
)
async def list_conversations(
    offset: int = 0,
    limit: int = 50,
    repo: PlaygroundRepository = Depends(get_playground_repo),
) -> PlaygroundConversationListResponse:
    """List playground conversations (most recent first)."""
    conversations, total = await repo.list_conversations(
        offset=offset, limit=limit
    )
    return PlaygroundConversationListResponse(
        conversations=[
            PlaygroundConversationResponse.model_validate(c)
            for c in conversations
        ],
        total=total,
    )


@router.get(
    "/conversations/{conversation_id}",
    response_model=PlaygroundConversationResponse,
)
async def get_conversation(
    conversation_id: uuid.UUID,
    repo: PlaygroundRepository = Depends(get_playground_repo),
) -> PlaygroundConversationResponse:
    """Get a single playground conversation."""
    conversation = await repo.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(404, "Conversation not found")
    return PlaygroundConversationResponse.model_validate(
        conversation
    )


@router.patch(
    "/conversations/{conversation_id}",
    response_model=PlaygroundConversationResponse,
)
async def update_conversation(
    conversation_id: uuid.UUID,
    body: UpdatePlaygroundConversationRequest,
    repo: PlaygroundRepository = Depends(get_playground_repo),
) -> PlaygroundConversationResponse:
    """Update conversation config (title, system_prompt, defenses)."""
    updates: dict[str, Any] = {}
    if body.title is not None:
        updates["title"] = body.title
    if body.system_prompt is not None:
        updates["system_prompt"] = body.system_prompt
    if body.active_defenses is not None:
        updates["active_defenses"] = [
            d.model_dump() for d in body.active_defenses
        ]
    if not updates:
        raise HTTPException(400, "No fields to update")

    conversation = await repo.update_conversation(
        conversation_id, **updates
    )
    if conversation is None:
        raise HTTPException(404, "Conversation not found")
    return PlaygroundConversationResponse.model_validate(
        conversation
    )


@router.delete(
    "/conversations/{conversation_id}",
    status_code=204,
)
async def delete_conversation(
    conversation_id: uuid.UUID,
    repo: PlaygroundRepository = Depends(get_playground_repo),
) -> None:
    """Delete a conversation and all its messages."""
    deleted = await repo.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(404, "Conversation not found")


# Messages


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=PlaygroundMessageListResponse,
)
async def list_messages(
    conversation_id: uuid.UUID,
    offset: int = 0,
    limit: int = 100,
    repo: PlaygroundRepository = Depends(get_playground_repo),
) -> PlaygroundMessageListResponse:
    """List messages in a playground conversation."""
    conversation = await repo.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(404, "Conversation not found")

    messages, total = await repo.list_messages(
        conversation_id, offset=offset, limit=limit
    )
    return PlaygroundMessageListResponse(
        messages=[
            PlaygroundMessageResponse.model_validate(m)
            for m in messages
        ],
        total=total,
    )


@router.post(
    "/conversations/{conversation_id}/messages",
    response_model=PlaygroundMessageResponse,
    status_code=201,
)
async def send_message(
    conversation_id: uuid.UUID,
    body: SendPlaygroundMessageRequest,
    repo: PlaygroundRepository = Depends(get_playground_repo),
    provider: OllamaProvider = Depends(get_ollama_provider),
) -> PlaygroundMessageResponse:
    """Process a user message through target + defenses + analyzer.

    This is synchronous as the caller waits for the full pipeline.
    WebSocket events provide phase-by-phase progress updates.
    """
    conversation = await repo.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(404, "Conversation not found")

    cid = str(conversation_id)
    msg_num = conversation.total_messages + 1

    # 1. Broadcast: started
    await broadcast_playground(cid, {
        "type": "pg_processing",
        "conversation_id": cid,
        "message_number": msg_num,
        "data": {"phase": "started"},
    })

    # 2. Instantiate defenses
    defenses = _instantiate_defenses(
        conversation.active_defenses, provider
    )

    # 3. Build TargetInterface
    target = TargetInterface(
        provider=provider,
        system_prompt=conversation.system_prompt,
        active_defenses=defenses,
    )

    # 4. Build conversation history from prior messages
    prior_messages, _ = await repo.list_messages(
        conversation_id, limit=100
    )
    history = _build_conversation_history(prior_messages)

    # 5. Build state dict for TargetInterface
    state: dict[str, Any] = {
        "current_attack_prompt": body.prompt,
        "target_model": conversation.target_model,
        "target_system_prompt": conversation.system_prompt,
        "conversation_history": history,
    }

    # 6. Call target LLM
    await broadcast_playground(cid, {
        "type": "pg_processing",
        "conversation_id": cid,
        "message_number": msg_num,
        "data": {"phase": "target_calling"},
    })

    t0 = time.perf_counter()
    target_result = await target(state)
    target_latency = (time.perf_counter() - t0) * 1000

    # 7. Extract defense block info
    blocked_by, block_reason = _parse_block_info(
        target_result.get("target_response", ""),
        target_result.get("target_blocked", False),
    )

    # 8. Broadcast: target responded
    await broadcast_playground(cid, {
        "type": "pg_processing",
        "conversation_id": cid,
        "message_number": msg_num,
        "data": {
            "phase": "target_responded",
            "target_response": target_result.get(
                "target_response", ""
            ),
            "target_blocked": target_result.get(
                "target_blocked", False
            ),
        },
    })

    # 9. Run analyzer pipeline
    await broadcast_playground(cid, {
        "type": "pg_processing",
        "conversation_id": cid,
        "message_number": msg_num,
        "data": {"phase": "analyzing"},
    })

    t1 = time.perf_counter()
    analysis = await _run_analysis(
        provider=provider,
        model=conversation.analyzer_model,
        attack_prompt=body.prompt,
        target_response=target_result.get(
            "target_response", ""
        ),
    )
    analyzer_latency = (time.perf_counter() - t1) * 1000

    # 10. Extract token counts
    target_tokens = _extract_target_tokens(target_result)
    analyzer_tokens = analysis.pop("_analyzer_tokens", 0)

    # 11. Persist message
    message = await repo.create_message(
        conversation_id=conversation_id,
        message_number=msg_num,
        user_prompt=body.prompt,
        target_response=target_result.get(
            "target_response", ""
        ),
        raw_target_response=target_result.get(
            "raw_target_response"
        ),
        target_blocked=target_result.get(
            "target_blocked", False
        ),
        blocked_by_defense=blocked_by,
        block_reason=block_reason,
        defenses_applied=conversation.active_defenses,
        judge_verdict=analysis.get(
            "judge_verdict", JudgeVerdict.ERROR
        ).value,
        judge_confidence=analysis.get(
            "judge_confidence", 0.0
        ),
        severity_score=analysis.get("severity_score"),
        specificity_score=analysis.get("specificity_score"),
        vulnerability_category=analysis.get(
            "vulnerability_category"
        ),
        attack_technique=analysis.get("attack_technique"),
        target_tokens=target_tokens,
        analyzer_tokens=analyzer_tokens,
        target_latency_ms=target_latency,
        analyzer_latency_ms=analyzer_latency,
    )

    # 12. Update conversation aggregate stats
    verdict = analysis.get(
        "judge_verdict", JudgeVerdict.ERROR
    )
    new_jailbreaks = (
        1 if verdict == JudgeVerdict.JAILBREAK else 0
    )
    new_blocked = (
        1 if target_result.get("target_blocked") else 0
    )
    await repo.update_conversation(
        conversation_id,
        total_messages=msg_num,
        total_jailbreaks=(
            conversation.total_jailbreaks + new_jailbreaks
        ),
        total_blocked=(
            conversation.total_blocked + new_blocked
        ),
        total_target_tokens=(
            conversation.total_target_tokens + target_tokens
        ),
        total_analyzer_tokens=(
            conversation.total_analyzer_tokens
            + analyzer_tokens
        ),
    )

    # 13. Broadcast: complete
    response = PlaygroundMessageResponse.model_validate(message)
    await broadcast_playground(cid, {
        "type": "pg_message_complete",
        "conversation_id": cid,
        "message_number": msg_num,
        "data": response.model_dump(mode="json"),
    })

    return response


# Helpers

_BLOCK_PATTERN = re.compile(
    r"\[(BLOCKED|OUTPUT FILTERED) by (\w+):\s*(.+?)\]"
)


def _parse_block_info(
    response: str, blocked: bool
) -> tuple[str | None, str | None]:
    """Extract defense name and reason from block messages."""
    if not blocked:
        return None, None
    match = _BLOCK_PATTERN.search(response)
    if match:
        return match.group(2), match.group(3)
    return None, None


def _build_conversation_history(
    messages: Any,
) -> list[dict[str, str]]:
    """Build chat history from prior PlaygroundMessages."""
    history: list[dict[str, str]] = []
    for msg in messages:
        history.append(
            {"role": "user", "content": msg.user_prompt}
        )
        # Use raw response if available (unblocked version)
        content = msg.raw_target_response or msg.target_response
        history.append(
            {"role": "assistant", "content": content}
        )
    return history


def _instantiate_defenses(
    defense_configs: list[dict[str, Any]],
    provider: BaseProvider,
) -> list[BaseDefense]:
    """Create defense instances from stored config dicts."""
    DefenseRegistry.discover()
    defenses: list[BaseDefense] = []
    for config in defense_configs:
        name = config.get("name", "")
        params = dict(config.get("params", {}))
        try:
            cls = DefenseRegistry.get(name)
            if name in _PROVIDER_DEFENSES:
                params["provider"] = provider
            defenses.append(cls(**params))
        except (KeyError, TypeError):
            logger.warning(
                "pg_defense_instantiation_failed", name=name
            )
    return defenses


async def _run_analysis(
    provider: BaseProvider,
    model: str,
    attack_prompt: str,
    target_response: str,
) -> dict[str, Any]:
    """Run judge + classifier + scorer pipeline directly."""
    total_tokens = 0

    # Step 1: Judge (always runs)
    judge_result = await run_judge(
        provider=provider,
        model=model,
        objective="[Playground - manual testing]",
        attack_prompt=attack_prompt,
        target_response=target_response,
    )

    # Extract judge tokens from token_budget
    token_budget = judge_result.pop("token_budget", None)
    if token_budget is not None:
        total_tokens += getattr(
            token_budget, "total_analyzer_tokens", 0
        )

    verdict = judge_result.get(
        "judge_verdict", JudgeVerdict.ERROR
    )
    result: dict[str, Any] = {**judge_result}

    # Step 2 & 3: Classifier + Scorer (jailbreak/borderline)
    if verdict in (
        JudgeVerdict.JAILBREAK,
        JudgeVerdict.BORDERLINE,
    ):
        classifier_result = await run_classifier(
            provider=provider,
            model=model,
            attack_prompt=attack_prompt,
            target_response=target_response,
            strategy_name="manual",
        )
        result.update(classifier_result)

        scorer_result = await run_scorer(
            provider=provider,
            model=model,
            target_response=target_response,
        )
        result.update(scorer_result)

    result["_analyzer_tokens"] = total_tokens
    return result


def _extract_target_tokens(
    target_result: dict[str, Any],
) -> int:
    """Extract target token count from TargetInterface result."""
    budget = target_result.get("token_budget")
    if budget is not None:
        return getattr(budget, "total_target_tokens", 0)
    return 0
