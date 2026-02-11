"""Abliteration dataset endpoints for managing prompt pairs."""

from __future__ import annotations

import json
import re
import uuid

import structlog
from fastapi import APIRouter, HTTPException, UploadFile

from adversarial_framework.agents.target.providers.ollama import (
    OllamaProvider,
)
from adversarial_framework.api.dependencies import (
    AbliterationPromptRepoDep,
    OllamaProviderDep,
)
from adversarial_framework.api.schemas.requests import (
    CreateAbliterationPromptRequest,
    GenerateHarmlessRequest,
    ModelLoadRequest,
    UpdateAbliterationPromptRequest,
)
from adversarial_framework.api.schemas.responses import (
    AbliterationDatasetStatsResponse,
    AbliterationPromptListResponse,
    AbliterationPromptResponse,
    DatasetSuggestionResponse,
)

logger = structlog.get_logger(__name__)
router = APIRouter()

_PROMPT_NOT_FOUND = "Prompt not found"
_MODEL_NOT_SELECTED = (
    "Please select and initialize a generation model first"
)

_HARMLESS_SYSTEM_PROMPT = (
    "Rewrite the following prompt as a harmless, educational request "
    "on a similar topic. Keep the same structure and length but remove "
    "any harmful intent. Do NOT include any reasoning, thinking, or "
    "explanation. Return ONLY the rewritten prompt, nothing else."
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


async def _generate_harmless_counterpart(
    harmful_text: str,
    provider: OllamaProvider,
    model: str,
) -> str:
    """Use the defender LLM to generate a topically similar harmless prompt."""
    messages = [
        {"role": "system", "content": _HARMLESS_SYSTEM_PROMPT},
        {"role": "user", "content": harmful_text},
    ]
    response = await provider.generate(messages=messages, model=model)
    text = _THINK_RE.sub("", response.content).strip()
    # Strip wrapping quotes if present
    if len(text) > 1 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()
    return text


# CRUD


@router.post(
    "/prompts",
    status_code=201,
)
async def add_prompt(
    body: CreateAbliterationPromptRequest,
    repo: AbliterationPromptRepoDep,
    provider: OllamaProviderDep,
    generation_model: str | None = None,
) -> AbliterationPromptResponse:
    """Add a single prompt to the dataset.

    If auto_generate_counterpart is True and category is harmful,
    also generates and saves a harmless counterpart using the
    model specified by the ``generation_model`` query param.
    """
    kwargs: dict[str, object] = {
        "text": body.text,
        "category": body.category,
        "source": body.source,
    }
    if body.experiment_id:
        kwargs["experiment_id"] = uuid.UUID(body.experiment_id)
    if body.session_id:
        kwargs["session_id"] = uuid.UUID(body.session_id)

    prompt = await repo.create(**kwargs)

    if (
        body.auto_generate_counterpart
        and body.category == "harmful"
    ):
        if not generation_model:
            raise HTTPException(
                status_code=400,
                detail=_MODEL_NOT_SELECTED,
            )
        try:
            harmless_text = await _generate_harmless_counterpart(
                body.text, provider, generation_model
            )
            await repo.create(
                text=harmless_text,
                category="harmless",
                source=body.source,
                pair_id=prompt.id,
                experiment_id=kwargs.get("experiment_id"),
                session_id=kwargs.get("session_id"),
            )
        except Exception:
            logger.warning(
                "harmless_generation_failed",
                harmful_prompt_id=str(prompt.id),
            )

    return AbliterationPromptResponse.model_validate(prompt)


@router.get("/prompts")
async def list_prompts(
    repo: AbliterationPromptRepoDep,
    offset: int = 0,
    limit: int = 50,
    category: str | None = None,
    source: str | None = None,
) -> AbliterationPromptListResponse:
    prompts, total = await repo.list_all(
        offset=offset,
        limit=limit,
        category=category,
        source=source,
    )
    return AbliterationPromptListResponse(
        prompts=[
            AbliterationPromptResponse.model_validate(p)
            for p in prompts
        ],
        total=total,
    )


@router.get("/prompts/stats")
async def get_stats(
    repo: AbliterationPromptRepoDep,
) -> AbliterationDatasetStatsResponse:
    counts = await repo.count_by_category()
    harmful = counts.get("harmful", 0)
    harmless = counts.get("harmless", 0)
    total = harmful + harmless
    pair_count = min(harmful, harmless)

    warning = None
    if pair_count < 16 and total > 0:
        warning = (
            f"Only {pair_count} pairs. "
            "Recommended minimum is 16 for reliable abliteration."
        )

    return AbliterationDatasetStatsResponse(
        harmful_count=harmful,
        harmless_count=harmless,
        total=total,
        warning=warning,
    )


@router.put("/prompts/{prompt_id}")
async def update_prompt(
    prompt_id: uuid.UUID,
    body: UpdateAbliterationPromptRequest,
    repo: AbliterationPromptRepoDep,
) -> AbliterationPromptResponse:
    prompt = await repo.update(
        prompt_id, text=body.text, category=body.category
    )
    if prompt is None:
        raise HTTPException(
            status_code=404, detail=_PROMPT_NOT_FOUND
        )
    return AbliterationPromptResponse.model_validate(prompt)


@router.delete("/prompts/{prompt_id}", status_code=204)
async def delete_prompt(
    prompt_id: uuid.UUID,
    repo: AbliterationPromptRepoDep,
) -> None:
    deleted = await repo.delete(prompt_id)
    if not deleted:
        raise HTTPException(
            status_code=404, detail=_PROMPT_NOT_FOUND
        )


# Upload


@router.post(
    "/prompts/upload",
    status_code=201,
)
async def upload_prompts(
    file: UploadFile,
    repo: AbliterationPromptRepoDep,
) -> AbliterationPromptListResponse:
    """Upload a JSONL file of prompt pairs.

    Each line: {"harmful": "...", "harmless": "..."}
    """
    if not file.filename or not file.filename.endswith(
        (".jsonl", ".json")
    ):
        raise HTTPException(
            status_code=400,
            detail="File must be .jsonl or .json",
        )

    content = await file.read()
    lines = content.decode("utf-8").strip().splitlines()

    to_create: list[dict[str, object]] = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            pair = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON on line {i + 1}: {exc}",
            ) from exc

        if "harmful" not in pair or "harmless" not in pair:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Line {i + 1}: each line must have "
                    "'harmful' and 'harmless' keys"
                ),
            )

        to_create.append(
            {
                "text": pair["harmful"],
                "category": "harmful",
                "source": "upload",
            }
        )
        to_create.append(
            {
                "text": pair["harmless"],
                "category": "harmless",
                "source": "upload",
            }
        )

    if not to_create:
        raise HTTPException(
            status_code=400, detail="No valid prompt pairs found"
        )

    created = await repo.create_many(to_create)
    # Link harmless→harmful pairs by position (persist to DB)
    for idx in range(0, len(created), 2):
        if idx + 1 < len(created):
            await repo.update(
                created[idx + 1].id,
                pair_id=created[idx].id,
            )

    return AbliterationPromptListResponse(
        prompts=[
            AbliterationPromptResponse.model_validate(p)
            for p in created
        ],
        total=len(created),
    )


# LLM Generation


@router.post(
    "/prompts/generate-harmless",
    status_code=201,
)
async def generate_harmless(
    body: GenerateHarmlessRequest,
    repo: AbliterationPromptRepoDep,
    provider: OllamaProviderDep,
    model: str | None = None,
) -> AbliterationPromptResponse:
    """Generate a harmless counterpart for a harmful prompt."""
    if not model:
        raise HTTPException(
            status_code=400,
            detail=_MODEL_NOT_SELECTED,
        )
    harmless_text = await _generate_harmless_counterpart(
        body.harmful_prompt, provider, model
    )

    kwargs: dict[str, object] = {
        "text": harmless_text,
        "category": "harmless",
        "source": "manual",
    }
    if body.pair_id:
        kwargs["pair_id"] = uuid.UUID(body.pair_id)
    if body.experiment_id:
        kwargs["experiment_id"] = uuid.UUID(body.experiment_id)
    if body.session_id:
        kwargs["session_id"] = uuid.UUID(body.session_id)

    prompt = await repo.create(**kwargs)
    return AbliterationPromptResponse.model_validate(prompt)


# Suggestions


@router.get("/prompts/suggestions")
async def list_suggestions(
    repo: AbliterationPromptRepoDep,
) -> DatasetSuggestionResponse:
    """List auto-detected suggestions from baseline refusals."""
    suggestions, total = await repo.list_suggestions()
    return DatasetSuggestionResponse(
        suggestions=[
            AbliterationPromptResponse.model_validate(s)
            for s in suggestions
        ],
        total=total,
    )


@router.post("/prompts/{prompt_id}/confirm")
async def confirm_suggestion(
    prompt_id: uuid.UUID,
    repo: AbliterationPromptRepoDep,
    provider: OllamaProviderDep,
    auto_generate_counterpart: bool = True,
    model: str | None = None,
) -> AbliterationPromptResponse:
    """Confirm a suggestion, moving it from 'suggested' to 'active'."""
    prompt = await repo.confirm(prompt_id)
    if prompt is None:
        raise HTTPException(
            status_code=404, detail=_PROMPT_NOT_FOUND
        )

    if auto_generate_counterpart:
        if not model:
            raise HTTPException(
                status_code=400,
                detail=_MODEL_NOT_SELECTED,
            )
        try:
            harmless_text = await _generate_harmless_counterpart(
                prompt.text, provider, model
            )
            await repo.create(
                text=harmless_text,
                category="harmless",
                source=prompt.source,
                pair_id=prompt.id,
                experiment_id=prompt.experiment_id,
                session_id=prompt.session_id,
            )
        except Exception:
            logger.warning(
                "harmless_generation_on_confirm_failed",
                prompt_id=str(prompt_id),
            )

    return AbliterationPromptResponse.model_validate(prompt)


@router.post(
    "/prompts/{prompt_id}/dismiss", status_code=204
)
async def dismiss_suggestion(
    prompt_id: uuid.UUID,
    repo: AbliterationPromptRepoDep,
) -> None:
    """Dismiss (delete) a suggestion."""
    deleted = await repo.delete(prompt_id)
    if not deleted:
        raise HTTPException(
            status_code=404, detail=_PROMPT_NOT_FOUND
        )


# Model Lifecycle


@router.post("/model/load", status_code=200)
async def load_model(
    body: ModelLoadRequest,
    provider: OllamaProviderDep,
) -> dict[str, str]:
    """Preload a model into Ollama memory for fast generation."""
    await provider.load_model(body.model)
    return {"status": "loaded", "model": body.model}


@router.post("/model/unload", status_code=200)
async def unload_model(
    body: ModelLoadRequest,
    provider: OllamaProviderDep,
) -> dict[str, str]:
    """Unload a model from Ollama memory to free resources."""
    await provider.unload_model(body.model)
    return {"status": "unloaded", "model": body.model}


@router.get("/model/status")
async def model_status(
    provider: OllamaProviderDep,
) -> dict[str, object]:
    """Return currently loaded Ollama models."""
    models = await provider.list_running_models()
    return {"models": models}
