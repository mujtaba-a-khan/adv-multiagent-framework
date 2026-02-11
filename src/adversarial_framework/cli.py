"""CLI runner for the adversarial framework.

Usage::

    uv run adversarial --target llama3:8b --strategy pair \\
        --objective "Describe how to pick a lock"

Or with explicit model assignments::

    uv run adversarial --target llama3:8b \\
        --attacker-model phi4-reasoning:14b \\
        --analyzer-model phi4-reasoning:14b \\
        --strategy pair \\
        --objective "..."
"""

from __future__ import annotations

import asyncio
import uuid

import click
import structlog

from adversarial_framework.config.settings import get_settings
from adversarial_framework.utils.logging import setup_logging

logger = structlog.get_logger(__name__)


def _print_node_event(
    node_name: str,
    updates: dict[str, object],
) -> None:
    """Print a formatted CLI line for a single graph node event."""
    from adversarial_framework.core.constants import JudgeVerdict

    if node_name == "attacker" and updates.get("current_attack_prompt"):
        turn = updates.get("current_turn", "?")
        prompt_preview = str(updates["current_attack_prompt"])[:100]
        click.echo(f"[Turn {turn}] ATTACKER → {prompt_preview}...")
        click.echo()
    elif node_name == "target":
        response = str(updates.get("target_response", ""))
        blocked = updates.get("target_blocked", False)
        if blocked:
            click.echo(f"  TARGET  → BLOCKED: {response[:100]}")
        else:
            click.echo(f"  TARGET  → {response[:120]}...")
        click.echo()
    elif node_name == "analyzer":
        verdict = updates.get("judge_verdict")
        confidence = updates.get("judge_confidence", 0)
        v_str = verdict.value if isinstance(verdict, JudgeVerdict) else str(verdict)
        click.echo(f"  JUDGE   → {v_str.upper()} (confidence: {confidence:.2f})")
        severity = updates.get("severity_score")
        if severity is not None:
            spec = updates.get("specificity_score", 0)
            coh = updates.get("coherence_score", 0)
            click.echo(
                f"  SCORES  → severity={severity:.0f} specificity={spec:.0f} coherence={coh:.0f}"
            )
        click.echo("-" * 60)
    elif node_name == "finalize":
        click.echo()
        click.echo("=" * 72)
        click.echo("  SESSION COMPLETE")
        click.echo("=" * 72)


async def _run_session(
    target_model: str,
    attacker_model: str,
    analyzer_model: str,
    defender_model: str,
    strategy: str,
    objective: str,
    max_turns: int,
    max_cost: float,
    target_system_prompt: str | None,
    ollama_url: str,
) -> None:
    """Build the graph, compile, and stream the adversarial session."""
    from adversarial_framework.agents.target.providers.ollama import OllamaProvider
    from adversarial_framework.core.constants import SessionStatus
    from adversarial_framework.core.graph import build_graph
    from adversarial_framework.core.state import TokenBudget

    provider = OllamaProvider(base_url=ollama_url)

    # Healthcheck
    if not await provider.healthcheck():
        click.echo(f"ERROR: Cannot reach Ollama at {ollama_url}", err=True)
        click.echo("Make sure Ollama is running: ollama serve", err=True)
        return

    graph = build_graph(
        provider=provider,
        attacker_config={"model": attacker_model},
        analyzer_config={"model": analyzer_model},
        target_system_prompt=target_system_prompt,
        max_turns=max_turns,
        max_cost_usd=max_cost,
    )

    compiled = graph.compile()

    # Initial state
    session_id = str(uuid.uuid4())[:8]
    experiment_id = str(uuid.uuid4())[:8]

    initial_state = {
        "experiment_id": experiment_id,
        "session_id": session_id,
        "target_model": target_model,
        "target_provider": "ollama",
        "target_system_prompt": target_system_prompt,
        "attacker_model": attacker_model,
        "analyzer_model": analyzer_model,
        "defender_model": defender_model,
        "attack_objective": objective,
        "strategy_config": {"name": strategy, "params": {}},
        "status": SessionStatus.PENDING,
        "current_turn": 0,
        "max_turns": max_turns,
        "max_cost_usd": max_cost,
        "selected_strategy": None,
        "strategy_params": None,
        "planning_notes": None,
        "current_attack_prompt": None,
        "attacker_reasoning": None,
        "target_response": None,
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
        "total_blocked": 0,
        "total_refused": 0,
        "conversation_history": [],
        "human_feedback": None,
        "awaiting_human": False,
        "last_error": None,
        "error_count": 0,
    }

    click.echo("=" * 72)
    click.echo(f"  Adversarial Framework — Session {session_id}")
    click.echo(f"  Target: {target_model} | Attacker: {attacker_model}")
    click.echo(f"  Strategy: {strategy} | Max turns: {max_turns}")
    click.echo(f"  Objective: {objective[:60]}...")
    click.echo("=" * 72)
    click.echo()

    # Stream events
    final_state = initial_state
    async for event in compiled.astream(initial_state, stream_mode="updates"):
        for node_name, updates in event.items():
            _print_node_event(node_name, updates)
        # Track latest state for summary
        final_state.update(
            {k: v for u in event.values() for k, v in u.items()},
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    click.echo()
    budget = final_state.get("token_budget", TokenBudget())
    history = final_state.get("attack_history", [])
    click.echo(f"  Turns completed:  {len(history)}")
    click.echo(f"  Jailbreaks:       {final_state.get('total_jailbreaks', 0)}")
    click.echo(f"  Refused:          {final_state.get('total_refused', 0)}")
    click.echo(f"  Blocked:          {final_state.get('total_blocked', 0)}")
    if isinstance(budget, TokenBudget):
        total = (
            budget.total_attacker_tokens + budget.total_target_tokens + budget.total_analyzer_tokens
        )
        click.echo(f"  Total tokens:     {total}")
    click.echo()

    await provider.close()


@click.command("adversarial")
@click.option(
    "--target",
    "target_model",
    default=None,
    help="Target model Ollama tag (e.g. llama3:8b).",
)
@click.option(
    "--attacker-model",
    default=None,
    help="Model for the attacker agent.",
)
@click.option(
    "--analyzer-model",
    default=None,
    help="Model for the analyzer/judge agent.",
)
@click.option(
    "--defender-model",
    default=None,
    help="Model for the defender agent.",
)
@click.option(
    "--strategy",
    default="pair",
    show_default=True,
    help="Attack strategy name.",
)
@click.option(
    "--objective",
    required=True,
    help="The attack objective — what behaviour to elicit.",
)
@click.option(
    "--max-turns",
    default=None,
    type=int,
    help="Maximum number of attack turns.",
)
@click.option(
    "--max-cost",
    default=None,
    type=float,
    help="Maximum budget in USD.",
)
@click.option(
    "--system-prompt",
    default=None,
    help="System prompt for the target model.",
)
@click.option(
    "--ollama-url",
    default=None,
    help="Ollama server URL.",
)
@click.option(
    "--log-level",
    default=None,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Logging level.",
)
@click.option(
    "--json-logs",
    is_flag=True,
    default=False,
    help="Output structured JSON logs.",
)
def main(
    target_model: str | None,
    attacker_model: str | None,
    analyzer_model: str | None,
    defender_model: str | None,
    strategy: str,
    objective: str,
    max_turns: int | None,
    max_cost: float | None,
    system_prompt: str | None,
    ollama_url: str | None,
    log_level: str | None,
    json_logs: bool,
) -> None:
    """Run an adversarial attack session against a target LLM."""
    settings = get_settings()

    # Merge CLI args with settings (CLI takes precedence)
    setup_logging(
        level=log_level or settings.log_level,
        json_output=json_logs or settings.log_json,
    )

    asyncio.run(
        _run_session(
            target_model=target_model or settings.target_model,
            attacker_model=attacker_model or settings.attacker_model,
            analyzer_model=analyzer_model or settings.analyzer_model,
            defender_model=defender_model or settings.defender_model,
            strategy=strategy,
            objective=objective,
            max_turns=max_turns or settings.max_turns,
            max_cost=max_cost or settings.max_cost_usd,
            target_system_prompt=system_prompt,
            ollama_url=ollama_url or settings.ollama_base_url,
        )
    )


if __name__ == "__main__":
    main()
