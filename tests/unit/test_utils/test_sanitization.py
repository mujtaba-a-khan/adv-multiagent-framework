"""Tests for adversarial_framework.utils.sanitization module."""

from __future__ import annotations

from adversarial_framework.utils.sanitization import (
    SanitizationResult,
    add_research_marker,
    normalize_encoding_artifacts,
    redact_pii,
    sanitize_for_storage,
    sanitize_log_data,
    strip_research_marker,
)

# SanitizationResult dataclass


class TestSanitizationResult:
    def test_frozen_dataclass(self):
        result = SanitizationResult(text="hello", modified=False, redactions=[])
        assert result.text == "hello"
        assert result.modified is False
        assert result.redactions == []

    def test_default_redactions(self):
        result = SanitizationResult(text="x", modified=True)
        assert result.redactions == []


# redact_pii


class TestRedactPii:
    def test_empty_string(self):
        result = redact_pii("")
        assert result.text == ""
        assert result.modified is False
        assert result.redactions == []

    def test_no_pii(self):
        result = redact_pii("Hello world, nothing to see here.")
        assert result.text == "Hello world, nothing to see here."
        assert result.modified is False

    def test_redacts_email(self):
        result = redact_pii("Contact user@example.com for info.")
        assert "[REDACTED]" in result.text
        assert "email" in result.redactions
        assert result.modified is True
        assert "user@example.com" not in result.text

    def test_redacts_phone_us(self):
        result = redact_pii("Call (555) 234-5678 now.")
        assert "[REDACTED]" in result.text
        assert "phone_us" in result.redactions
        assert result.modified is True

    def test_redacts_ssn(self):
        result = redact_pii("SSN: 123-45-6789")
        assert "[REDACTED]" in result.text
        assert "ssn" in result.redactions
        assert result.modified is True
        assert "123-45-6789" not in result.text

    def test_redacts_credit_card(self):
        result = redact_pii("Card: 4111-1111-1111-1111")
        assert "[REDACTED]" in result.text
        assert "credit_card" in result.redactions
        assert result.modified is True

    def test_redacts_ip_address(self):
        result = redact_pii("Server at 192.168.1.1 is down.")
        assert "[REDACTED]" in result.text
        assert "ip_address" in result.redactions
        assert result.modified is True
        assert "192.168.1.1" not in result.text

    def test_custom_replacement(self):
        result = redact_pii("Email: a@b.com", replacement="***")
        assert "***" in result.text
        assert "a@b.com" not in result.text

    def test_multiple_pii_types(self):
        text = "Email: a@b.com, IP: 10.0.0.1"
        result = redact_pii(text)
        assert "email" in result.redactions
        assert "ip_address" in result.redactions
        assert result.modified is True

    def test_multiple_same_type(self):
        text = "a@b.com and c@d.com"
        result = redact_pii(text)
        assert result.text.count("[REDACTED]") == 2
        assert result.redactions.count("email") == 1


# normalize_encoding_artifacts


class TestNormalizeEncodingArtifacts:
    def test_plain_text_unchanged(self):
        assert normalize_encoding_artifacts("hello") == "hello"

    def test_removes_zero_width_space(self):
        assert normalize_encoding_artifacts("he\u200bllo") == "hello"

    def test_removes_zero_width_joiner(self):
        assert normalize_encoding_artifacts("he\u200dllo") == "hello"

    def test_removes_bom(self):
        assert normalize_encoding_artifacts("\ufeffhello") == "hello"

    def test_removes_soft_hyphen(self):
        assert normalize_encoding_artifacts("he\u00adllo") == "hello"

    def test_removes_multiple_invisible(self):
        text = "\u200b\u200c\u200d\u200e\u200f\ufefftest"
        assert normalize_encoding_artifacts(text) == "test"

    def test_empty_string(self):
        assert normalize_encoding_artifacts("") == ""


# add_research_marker / strip_research_marker


class TestResearchMarker:
    def test_add_marker(self):
        result = add_research_marker("some data")
        assert result.startswith("[ADVERSARIAL RESEARCH DATA")
        assert "some data" in result

    def test_add_marker_idempotent(self):
        text = "test content"
        once = add_research_marker(text)
        twice = add_research_marker(once)
        assert once == twice

    def test_strip_marker(self):
        marked = add_research_marker("content")
        stripped = strip_research_marker(marked)
        assert stripped == "content"

    def test_strip_no_marker(self):
        assert strip_research_marker("plain text") == "plain text"

    def test_roundtrip(self):
        original = "sensitive data"
        assert strip_research_marker(add_research_marker(original)) == original


# sanitize_for_storage
class TestSanitizeForStorage:
    def test_empty_string(self):
        result = sanitize_for_storage("")
        assert result.text == ""
        assert result.modified is False

    def test_no_modifications_needed(self):
        result = sanitize_for_storage("clean text")
        assert result.text == "clean text"
        assert result.modified is False

    def test_normalize_only(self):
        result = sanitize_for_storage("he\u200bllo", redact=False, normalize=True)
        assert result.text == "hello"
        assert result.modified is True

    def test_redact_only(self):
        result = sanitize_for_storage("a@b.com", redact=True, normalize=False)
        assert "[REDACTED]" in result.text
        assert result.modified is True
        assert "email" in result.redactions

    def test_mark_as_research(self):
        result = sanitize_for_storage("data", mark_as_research=True)
        assert result.text.startswith("[ADVERSARIAL RESEARCH DATA")
        assert result.modified is True

    def test_all_options(self):
        text = "he\u200bllo a@b.com"
        result = sanitize_for_storage(
            text,
            redact=True,
            normalize=True,
            mark_as_research=True,
        )
        assert result.modified is True
        assert "email" in result.redactions
        assert "\u200b" not in result.text
        assert result.text.startswith("[ADVERSARIAL RESEARCH DATA")

    def test_no_redact_no_normalize(self):
        result = sanitize_for_storage(
            "a@b.com \u200b",
            redact=False,
            normalize=False,
        )
        assert result.text == "a@b.com \u200b"
        assert result.modified is False


# sanitize_log_data


class TestSanitizeLogData:
    def test_empty_dict(self):
        assert sanitize_log_data({}) == {}

    def test_non_string_values_pass_through(self):
        data = {"count": 42, "flag": True, "items": [1, 2]}
        assert sanitize_log_data(data) == data

    def test_truncates_long_strings(self):
        data = {"msg": "a" * 600}
        result = sanitize_log_data(data, max_length=100)
        assert len(result["msg"]) < 600
        assert "600 chars total" in result["msg"]

    def test_redacts_pii_in_strings(self):
        data = {"email": "Contact user@example.com"}
        result = sanitize_log_data(data)
        assert "user@example.com" not in result["email"]
        assert "[REDACTED]" in result["email"]

    def test_nested_dict(self):
        data = {"inner": {"email": "a@b.com"}}
        result = sanitize_log_data(data)
        assert "[REDACTED]" in result["inner"]["email"]

    def test_custom_max_length(self):
        data = {"msg": "a" * 20}
        result = sanitize_log_data(data, max_length=10)
        assert "20 chars total" in result["msg"]

    def test_string_under_limit_not_truncated(self):
        data = {"msg": "short"}
        result = sanitize_log_data(data, max_length=500)
        assert result["msg"] == "short"
