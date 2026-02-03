"""Content safety sanitization utilities.

Provides input/output sanitization for adversarial research data stored
in the database or displayed in the UI. Handles PII detection, encoding
normalization, and content safety markers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SanitizationResult:
    """Result of sanitizing a text string."""

    text: str
    modified: bool
    redactions: list[str] = field(default_factory=list)


# PII patterns (conservative to avoid false positives)
_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone_us": re.compile(
        r"\b(?:\+?1[-.\s]?)?(?:\(?[2-9]\d{2}\)?[-.\s]?)[2-9]\d{2}[-.\s]?\d{4}\b"
    ),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "ip_address": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ),
}

# Content safety markers
_RESEARCH_HEADER = "[ADVERSARIAL RESEARCH DATA — NOT FOR PRODUCTION USE]"


def redact_pii(text: str, replacement: str = "[REDACTED]") -> SanitizationResult:
    """Detect and redact PII patterns from text.

    Args:
        text: Input text to sanitize.
        replacement: String to replace PII matches with.

    Returns:
        SanitizationResult with redacted text and list of redaction types.
    """
    if not text:
        return SanitizationResult(text="", modified=False)

    result = text
    redactions: list[str] = []

    for pii_type, pattern in _PII_PATTERNS.items():
        if pattern.search(result):
            result = pattern.sub(replacement, result)
            redactions.append(pii_type)

    return SanitizationResult(
        text=result,
        modified=bool(redactions),
        redactions=redactions,
    )


def normalize_encoding_artifacts(text: str) -> str:
    """Remove zero-width and invisible Unicode characters.

    These are sometimes used in adversarial prompts to bypass filters.
    This normalisation makes stored data safer for analysis.
    """
    # Zero-width characters
    invisible = re.compile(
        r"[\u200b\u200c\u200d\u200e\u200f"
        r"\u2060\u2061\u2062\u2063\u2064"
        r"\ufeff\u00ad\u034f\u17b4\u17b5"
        r"\u180b\u180c\u180d\u180e\u180f]"
    )
    return invisible.sub("", text)


def add_research_marker(text: str) -> str:
    """Prepend a research data marker to stored content."""
    if text.startswith(_RESEARCH_HEADER):
        return text
    return f"{_RESEARCH_HEADER}\n{text}"


def strip_research_marker(text: str) -> str:
    """Remove research data marker for display."""
    if text.startswith(_RESEARCH_HEADER):
        return text[len(_RESEARCH_HEADER) :].lstrip("\n")
    return text


def sanitize_for_storage(
    text: str,
    redact: bool = True,
    normalize: bool = True,
    mark_as_research: bool = False,
) -> SanitizationResult:
    """Apply full sanitization pipeline for database storage.

    Args:
        text: Raw text to sanitize.
        redact: Whether to redact PII.
        normalize: Whether to normalize encoding artifacts.
        mark_as_research: Whether to add research data marker.

    Returns:
        SanitizationResult with sanitized text.
    """
    if not text:
        return SanitizationResult(text="", modified=False)

    result = text
    all_redactions: list[str] = []
    modified = False

    if normalize:
        normalized = normalize_encoding_artifacts(result)
        if normalized != result:
            modified = True
            result = normalized

    if redact:
        pii_result = redact_pii(result)
        if pii_result.modified:
            modified = True
            result = pii_result.text
            all_redactions.extend(pii_result.redactions)

    if mark_as_research:
        marked = add_research_marker(result)
        if marked != result:
            modified = True
            result = marked

    return SanitizationResult(
        text=result, modified=modified, redactions=all_redactions
    )


def sanitize_log_data(data: dict[str, Any], max_length: int = 500) -> dict[str, Any]:
    """Sanitize a dict for safe logging (truncate long values, redact PII).

    Args:
        data: Dictionary of log data.
        max_length: Maximum character length for string values.

    Returns:
        Sanitized copy of the dict.
    """
    sanitized: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str):
            # Truncate
            if len(value) > max_length:
                value = value[:max_length] + f"... [{len(value)} chars total]"
            # Redact PII
            result = redact_pii(value)
            sanitized[key] = result.text
        elif isinstance(value, dict):
            sanitized[key] = sanitize_log_data(value, max_length)
        else:
            sanitized[key] = value
    return sanitized
