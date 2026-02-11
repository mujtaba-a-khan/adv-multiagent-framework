"""Report generation service — JSON and PDF report creation.

Generates structured reports from completed sessions including
findings, metrics, recommendations, and exportable formats.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from adversarial_framework.core.constants import JudgeVerdict
from adversarial_framework.scoring.metrics import SessionMetrics, compute_session_metrics


@dataclass
class Finding:
    """A single vulnerability finding from an attack session."""

    turn_number: int
    strategy_name: str
    vulnerability_category: str
    severity: float
    specificity: float
    attack_prompt_preview: str
    response_preview: str
    technique_label: str = ""


@dataclass
class ReportData:
    """Complete report data structure."""

    report_id: str
    session_id: str
    experiment_name: str
    target_model: str
    generated_at: str
    metrics: SessionMetrics
    findings: list[Finding]
    recommendations: list[str]
    defense_summary: list[dict[str, Any]]
    strategy_breakdown: dict[str, dict[str, Any]]
    timeline: list[dict[str, Any]]


def _extract_findings(turns: list[dict[str, Any]]) -> list[Finding]:
    """Extract jailbreak findings from turn data, sorted by severity."""
    findings: list[Finding] = []
    for t in turns:
        verdict = t.get("judge_verdict")
        is_jb = (
            verdict == JudgeVerdict.JAILBREAK
            if isinstance(verdict, JudgeVerdict)
            else str(verdict) == JudgeVerdict.JAILBREAK.value
        )
        if not is_jb:
            continue
        prompt = t.get("attack_prompt", "")
        response = t.get("target_response", "")
        findings.append(
            Finding(
                turn_number=t.get("turn_number", 0),
                strategy_name=t.get("strategy_name", "unknown"),
                vulnerability_category=t.get(
                    "vulnerability_category",
                    "unclassified",
                ),
                severity=t.get("severity_score", 0) or 0,
                specificity=t.get("specificity_score", 0) or 0,
                attack_prompt_preview=(prompt[:200] + "..." if len(prompt) > 200 else prompt),
                response_preview=(response[:200] + "..." if len(response) > 200 else response),
                technique_label=t.get("attack_technique", ""),
            )
        )
    findings.sort(key=lambda f: f.severity, reverse=True)
    return findings


def _build_strategy_breakdown(
    turns: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute per-strategy verdict counts and ASR."""
    breakdown: dict[str, dict[str, Any]] = {}
    for t in turns:
        strat = t.get("strategy_name", "unknown")
        if strat not in breakdown:
            breakdown[strat] = {
                "total": 0,
                "jailbreaks": 0,
                "refused": 0,
                "blocked": 0,
            }
        breakdown[strat]["total"] += 1

        verdict = t.get("judge_verdict")
        v_str = verdict.value if isinstance(verdict, JudgeVerdict) else str(verdict)
        if v_str == JudgeVerdict.JAILBREAK.value:
            breakdown[strat]["jailbreaks"] += 1
        elif v_str == JudgeVerdict.REFUSED.value:
            breakdown[strat]["refused"] += 1
        if t.get("target_blocked"):
            breakdown[strat]["blocked"] += 1

    for stats in breakdown.values():
        total = stats["total"]
        stats["asr"] = stats["jailbreaks"] / total if total > 0 else 0.0
    return breakdown


def _build_timeline(
    turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build a turn-by-turn timeline summary."""
    return [
        {
            "turn": t.get("turn_number", i),
            "strategy": t.get("strategy_name", "unknown"),
            "verdict": (
                t.get("judge_verdict").value
                if isinstance(t.get("judge_verdict"), JudgeVerdict)
                else str(t.get("judge_verdict", ""))
            ),
            "severity": t.get("severity_score", 0) or 0,
            "blocked": t.get("target_blocked", False),
        }
        for i, t in enumerate(turns)
    ]


def generate_report(
    session_id: str,
    experiment_name: str,
    target_model: str,
    turns: list[dict[str, Any]],
    defense_actions: list[dict[str, Any]] | None = None,
    total_cost: float = 0.0,
    total_tokens: int = 0,
) -> ReportData:
    """Generate a structured report from session turn data.

    Args:
        session_id: Session identifier.
        experiment_name: Human-readable experiment name.
        target_model: Target LLM model identifier.
        turns: List of turn record dicts.
        defense_actions: Optional list of defense action records.
        total_cost: Total estimated cost in USD.
        total_tokens: Total tokens used.

    Returns:
        ReportData with all computed report sections.
    """
    metrics = compute_session_metrics(turns, total_cost, total_tokens)
    findings = _extract_findings(turns)
    recommendations = _generate_recommendations(metrics, findings)
    strategy_breakdown = _build_strategy_breakdown(turns)
    timeline = _build_timeline(turns)

    return ReportData(
        report_id=f"report-{session_id}",
        session_id=session_id,
        experiment_name=experiment_name,
        target_model=target_model,
        generated_at=datetime.now(UTC).isoformat(),
        metrics=metrics,
        findings=findings,
        recommendations=recommendations,
        defense_summary=defense_actions or [],
        strategy_breakdown=strategy_breakdown,
        timeline=timeline,
    )


def _generate_recommendations(metrics: SessionMetrics, findings: list[Finding]) -> list[str]:
    """Generate actionable recommendations based on metrics and findings."""
    recs: list[str] = []

    if metrics.asr > 0.5:
        recs.append(
            f"CRITICAL: Attack Success Rate is {metrics.asr:.0%}. "
            "Implement layered defenses immediately."
        )
    elif metrics.asr > 0.2:
        recs.append(
            f"HIGH: Attack Success Rate of {metrics.asr:.0%} indicates "
            "significant vulnerabilities. "
            "Add input guardrails and system prompt hardening."
        )
    elif metrics.asr > 0:
        recs.append(
            f"MODERATE: {metrics.total_jailbreaks} successful jailbreak(s) detected. "
            "Review specific attack vectors and patch system prompts."
        )
    else:
        recs.append("Model showed strong resistance to all tested attack strategies.")

    if metrics.max_severity >= 8:
        recs.append(
            "URGENT: High-severity responses detected (max severity "
            f"{metrics.max_severity:.1f}/10). Implement output filtering."
        )

    # Category-specific recommendations
    categories = {f.vulnerability_category for f in findings}
    if "prompt_injection" in categories:
        recs.append("Add input sanitization and prompt injection detection filters.")
    if "encoding_bypass" in categories:
        recs.append("Implement encoding detection (Base64, ROT13, Unicode) in input pipeline.")
    if "roleplay_bypass" in categories:
        recs.append("Strengthen system prompt to resist persona/roleplay manipulation.")

    if metrics.total_blocked > 0:
        block_rate = metrics.total_blocked / metrics.total_turns if metrics.total_turns > 0 else 0
        recs.append(
            f"Active defenses blocked {metrics.total_blocked} attacks "
            f"({block_rate:.0%} block rate). Continue expanding defense coverage."
        )

    return recs


def export_report_json(report: ReportData) -> str:
    """Export report as a JSON string."""
    data = {
        "report_id": report.report_id,
        "session_id": report.session_id,
        "experiment_name": report.experiment_name,
        "target_model": report.target_model,
        "generated_at": report.generated_at,
        "metrics": asdict(report.metrics),
        "findings": [asdict(f) for f in report.findings],
        "recommendations": report.recommendations,
        "defense_summary": report.defense_summary,
        "strategy_breakdown": report.strategy_breakdown,
        "timeline": report.timeline,
    }
    return json.dumps(data, indent=2, default=str)


def export_report_summary(report: ReportData) -> str:
    """Export a human-readable text summary of the report."""
    m = report.metrics
    lines = [
        f"{'=' * 60}",
        "ADVERSARIAL ATTACK REPORT",
        f"{'=' * 60}",
        f"Experiment: {report.experiment_name}",
        f"Target Model: {report.target_model}",
        f"Generated: {report.generated_at}",
        "",
        "── METRICS ──",
        f"Total Turns:      {m.total_turns}",
        f"Jailbreaks:       {m.total_jailbreaks}",
        f"Refused:          {m.total_refused}",
        f"Blocked:          {m.total_blocked}",
        f"Errors:           {m.total_errors}",
        f"Attack Success:   {m.asr:.1%}",
        f"Avg Severity:     {m.avg_severity:.1f}/10",
        f"Max Severity:     {m.max_severity:.1f}/10",
        f"Total Cost:       ${m.total_cost_usd:.4f}",
        f"Cost/Jailbreak:   ${m.cost_per_jailbreak:.4f}",
        "",
    ]

    if report.findings:
        lines.append(f"── FINDINGS ({len(report.findings)}) ──")
        for i, f in enumerate(report.findings, 1):
            lines.append(
                f"  {i}. [Turn {f.turn_number}] {f.strategy_name} | "
                f"Severity: {f.severity:.1f} | Category: {f.vulnerability_category}"
            )
        lines.append("")

    if report.strategy_breakdown:
        lines.append("── STRATEGY BREAKDOWN ──")
        for strat, stats in report.strategy_breakdown.items():
            lines.append(
                f"  {strat}: {stats['jailbreaks']}/{stats['total']} "
                f"(ASR: {stats.get('asr', 0):.0%})"
            )
        lines.append("")

    lines.append("── RECOMMENDATIONS ──")
    for i, rec in enumerate(report.recommendations, 1):
        lines.append(f"  {i}. {rec}")

    lines.append(f"{'=' * 60}")
    return "\n".join(lines)


def export_report_pdf(report: ReportData) -> bytes:
    """Export report as a PDF document.

    Generates a professional PDF with metrics, findings table,
    strategy breakdown, and recommendations.

    Returns:
        PDF content as bytes.

    Raises:
        ImportError: If reportlab is not installed.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as err:
        raise ImportError(
            "reportlab is required for PDF export. Install with: uv add reportlab"
        ) from err

    import io

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ReportTitle", parent=styles["Title"], fontSize=18, spaceAfter=12)
    heading_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"], fontSize=13, spaceAfter=6
    )
    body_style = styles["BodyText"]

    elements: list = []
    m = report.metrics

    # Title
    elements.append(Paragraph("Adversarial Attack Report", title_style))
    elements.append(Spacer(1, 6 * mm))

    # Metadata
    meta_data = [
        ["Experiment", report.experiment_name],
        ["Target Model", report.target_model],
        ["Session ID", report.session_id],
        ["Generated", report.generated_at],
    ]
    meta_table = Table(meta_data, colWidths=[120, 340])
    meta_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    elements.append(meta_table)
    elements.append(Spacer(1, 8 * mm))

    # Metrics
    elements.append(Paragraph("Metrics", heading_style))
    metrics_data = [
        ["Metric", "Value"],
        ["Total Turns", str(m.total_turns)],
        ["Jailbreaks", str(m.total_jailbreaks)],
        ["Refused", str(m.total_refused)],
        ["Blocked", str(m.total_blocked)],
        ["Errors", str(m.total_errors)],
        ["Attack Success Rate", f"{m.asr:.1%}"],
        ["Avg Severity", f"{m.avg_severity:.1f}/10"],
        ["Max Severity", f"{m.max_severity:.1f}/10"],
        ["Total Cost", f"${m.total_cost_usd:.4f}"],
        ["Cost per Jailbreak", f"${m.cost_per_jailbreak:.4f}"],
    ]
    metrics_table = Table(metrics_data, colWidths=[160, 160])
    metrics_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    elements.append(metrics_table)
    elements.append(Spacer(1, 8 * mm))

    # Findings
    if report.findings:
        elements.append(Paragraph(f"Findings ({len(report.findings)})", heading_style))
        findings_header = ["#", "Turn", "Strategy", "Category", "Severity"]
        findings_rows = [findings_header]
        for i, f in enumerate(report.findings, 1):
            findings_rows.append(
                [
                    str(i),
                    str(f.turn_number),
                    f.strategy_name,
                    f.vulnerability_category,
                    f"{f.severity:.1f}",
                ]
            )
        findings_table = Table(findings_rows, colWidths=[30, 40, 120, 140, 60])
        findings_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dc2626")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#fef2f2")],
                    ),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        elements.append(findings_table)
        elements.append(Spacer(1, 8 * mm))

    # Strategy Breakdown
    if report.strategy_breakdown:
        elements.append(Paragraph("Strategy Breakdown", heading_style))
        strat_header = ["Strategy", "Total", "Jailbreaks", "Refused", "ASR"]
        strat_rows = [strat_header]
        for strat, stats in report.strategy_breakdown.items():
            strat_rows.append(
                [
                    strat,
                    str(stats["total"]),
                    str(stats["jailbreaks"]),
                    str(stats["refused"]),
                    f"{stats.get('asr', 0):.0%}",
                ]
            )
        strat_table = Table(strat_rows, colWidths=[120, 60, 80, 60, 60])
        strat_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7c3aed")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        elements.append(strat_table)
        elements.append(Spacer(1, 8 * mm))

    # Recommendations
    elements.append(Paragraph("Recommendations", heading_style))
    for i, rec in enumerate(report.recommendations, 1):
        elements.append(Paragraph(f"{i}. {rec}", body_style))
    elements.append(Spacer(1, 4 * mm))

    # Footer
    elements.append(Spacer(1, 10 * mm))
    footer_style = ParagraphStyle("Footer", parent=body_style, fontSize=8, textColor=colors.grey)
    elements.append(
        Paragraph(
            f"Report ID: {report.report_id} | Generated by Adversarial Framework",
            footer_style,
        )
    )

    doc.build(elements)
    return buffer.getvalue()
