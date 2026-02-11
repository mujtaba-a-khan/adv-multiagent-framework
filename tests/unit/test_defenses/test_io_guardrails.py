"""Tests for IOGuardrailsDefense."""

from __future__ import annotations

from adversarial_framework.defenses.io_guardrails import IOGuardrailsDefense


class TestIOGuardrailsDefenseName:
    def test_name(self) -> None:
        defense = IOGuardrailsDefense()
        assert defense.name == "io_guardrails"


class TestIOGuardrailsDefenseCheckInput:
    """Input check always passes through (I/O guardrails protect outputs)."""

    async def test_input_always_passes(self) -> None:
        defense = IOGuardrailsDefense()
        result = await defense.check_input("Any prompt at all")
        assert result.blocked is False

    async def test_input_with_pii_still_passes(self) -> None:
        defense = IOGuardrailsDefense()
        result = await defense.check_input("My SSN is 123-45-6789")
        assert result.blocked is False

    async def test_input_with_email_passes(self) -> None:
        defense = IOGuardrailsDefense()
        result = await defense.check_input("Contact me at test@example.com")
        assert result.blocked is False


class TestIOGuardrailsDefenseOutputSSN:
    """SSN detection in output."""

    async def test_ssn_detected_and_blocked(self) -> None:
        defense = IOGuardrailsDefense()
        result = await defense.check_output("The user's SSN is 123-45-6789.")
        assert result.blocked is True
        assert "ssn" in result.reason
        assert result.confidence == 0.90

    async def test_ssn_metadata(self) -> None:
        defense = IOGuardrailsDefense()
        result = await defense.check_output("SSN: 999-88-7777")
        assert result.metadata is not None
        pii = result.metadata["pii_detected"]
        pii_types = [d["type"] for d in pii]
        assert "ssn" in pii_types


class TestIOGuardrailsDefenseOutputCreditCard:
    """Credit card detection in output."""

    async def test_credit_card_detected(self) -> None:
        defense = IOGuardrailsDefense()
        result = await defense.check_output("Card number: 4111-1111-1111-1111")
        assert result.blocked is True
        assert "credit_card" in result.reason


class TestIOGuardrailsDefenseOutputEmail:
    """Email detection in output."""

    async def test_email_detected(self) -> None:
        defense = IOGuardrailsDefense()
        result = await defense.check_output("Send mail to john.doe@example.com for details.")
        assert result.blocked is True
        assert "email" in result.reason


class TestIOGuardrailsDefenseOutputPhone:
    """Phone number detection in output."""

    async def test_us_phone_detected(self) -> None:
        defense = IOGuardrailsDefense()
        result = await defense.check_output("Call us at (555) 123-4567.")
        assert result.blocked is True
        assert "phone_us" in result.reason


class TestIOGuardrailsDefenseOutputIP:
    """IP address detection in output."""

    async def test_ip_address_detected(self) -> None:
        defense = IOGuardrailsDefense()
        result = await defense.check_output("The server is at 192.168.1.100.")
        assert result.blocked is True
        assert "ip_address" in result.reason


class TestIOGuardrailsDefenseOutputAPIKey:
    """API key detection in output."""

    async def test_openai_api_key_detected(self) -> None:
        defense = IOGuardrailsDefense()
        fake_key = "sk-" + "a" * 30
        result = await defense.check_output(f"Use this API key: {fake_key}")
        assert result.blocked is True
        assert "api_key" in result.reason

    async def test_aws_access_key_detected(self) -> None:
        defense = IOGuardrailsDefense()
        result = await defense.check_output("AWS key: AKIAIOSFODNN7EXAMPLE")
        assert result.blocked is True
        assert "api_key" in result.reason


class TestIOGuardrailsDefenseOutputClean:
    """Clean outputs without PII should pass."""

    async def test_clean_output_passes(self) -> None:
        defense = IOGuardrailsDefense()
        result = await defense.check_output("The capital of France is Paris.")
        assert result.blocked is False

    async def test_numbers_not_false_positive(self) -> None:
        defense = IOGuardrailsDefense()
        result = await defense.check_output("The answer is 42.")
        assert result.blocked is False


class TestIOGuardrailsDefenseMultiplePII:
    """Multiple PII types in one response."""

    async def test_multiple_pii_types_in_reason(self) -> None:
        defense = IOGuardrailsDefense()
        result = await defense.check_output("SSN: 123-45-6789, email: user@test.com")
        assert result.blocked is True
        assert "ssn" in result.reason
        assert "email" in result.reason


class TestIOGuardrailsDefenseBlockPIIFlag:
    """block_pii_in_output=False should detect but not block."""

    async def test_pii_detected_but_not_blocked(self) -> None:
        defense = IOGuardrailsDefense(block_pii_in_output=False)
        result = await defense.check_output("SSN: 123-45-6789")
        assert result.blocked is False


class TestIOGuardrailsDefenseCustomPatterns:
    """Custom PII patterns can extend detection."""

    async def test_custom_pattern_detected(self) -> None:
        defense = IOGuardrailsDefense(custom_pii_patterns=[("passport", r"\b[A-Z]{2}\d{7}\b")])
        result = await defense.check_output("Passport number: AB1234567")
        assert result.blocked is True
        assert "passport" in result.reason

    async def test_custom_pattern_no_false_positive(self) -> None:
        defense = IOGuardrailsDefense(custom_pii_patterns=[("passport", r"\b[A-Z]{2}\d{7}\b")])
        result = await defense.check_output("This is a clean response.")
        assert result.blocked is False

    async def test_custom_pattern_combines_with_builtin(self) -> None:
        defense = IOGuardrailsDefense(custom_pii_patterns=[("passport", r"\b[A-Z]{2}\d{7}\b")])
        result = await defense.check_output("Passport: AB1234567, SSN: 123-45-6789")
        assert result.blocked is True
        assert "passport" in result.reason
        assert "ssn" in result.reason
