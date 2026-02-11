"""Tests for the memory estimation and feasibility module."""

from __future__ import annotations

from unittest.mock import patch

from adversarial_framework.services.finetuning.memory import (
    check_memory_feasibility,
    estimate_memory_gb,
    get_available_memory_gb,
)


class TestEstimateMemoryGB:
    def test_pull_always_small(self):
        """Pull jobs only download — memory usage is negligible."""
        result = estimate_memory_gb("llama3:70b", "pull_abliterated")
        assert result == 0.5

    def test_8b_abliterate(self):
        """8B model abliteration should use ~6.5GB (5.0 * 1.3)."""
        result = estimate_memory_gb("llama3:8b", "abliterate")
        assert result == 5.0 * 1.3

    def test_8b_sft(self):
        """8B model SFT should use ~9GB (5.0 * 1.8)."""
        result = estimate_memory_gb("llama3:8b", "sft")
        assert result == 5.0 * 1.8

    def test_7b_model_detection(self):
        result = estimate_memory_gb("mistral:7b-instruct", "abliterate")
        assert result == 4.5 * 1.3

    def test_14b_model_detection(self):
        result = estimate_memory_gb("phi4-reasoning:14b", "abliterate")
        assert result == 8.5 * 1.3

    def test_1b_model_detection(self):
        result = estimate_memory_gb("qwen2:1b", "sft")
        assert result == 0.8 * 1.8

    def test_unknown_model_uses_default(self):
        """Unknown model size defaults to 5.0GB base."""
        result = estimate_memory_gb("some-custom-model", "abliterate")
        assert result == 5.0 * 1.3

    def test_unknown_job_type_returns_base(self):
        result = estimate_memory_gb("llama3:8b", "unknown_type")
        assert result == 5.0

    def test_case_insensitive_model_tag(self):
        result = estimate_memory_gb("LLaMA3:8B", "abliterate")
        assert result == 5.0 * 1.3


class TestGetAvailableMemoryGB:
    def test_returns_float(self):
        result = get_available_memory_gb()
        assert isinstance(result, float)
        assert result > 0

    def test_fallback_without_psutil(self):
        with (
            patch.dict("sys.modules", {"psutil": None}),
            patch(
                "adversarial_framework.services.finetuning.memory.get_available_memory_gb",
                return_value=16.0,
            ),
        ):
            assert get_available_memory_gb() == 16.0


class TestCheckMemoryFeasibility:
    def test_pull_always_feasible(self):
        feasible, reason = check_memory_feasibility("llama3:70b", "pull_abliterated")
        assert feasible is True
        assert reason == ""

    def test_small_model_feasible(self):
        """1B model should fit on any system."""
        with patch(
            "adversarial_framework.services.finetuning.memory.get_available_memory_gb",
            return_value=16.0,
        ):
            feasible, reason = check_memory_feasibility("qwen2:1b", "abliterate")
            assert feasible is True

    def test_huge_model_infeasible(self):
        """70B model should not fit on 16GB."""
        with patch(
            "adversarial_framework.services.finetuning.memory.get_available_memory_gb",
            return_value=16.0,
        ):
            feasible, reason = check_memory_feasibility("llama3:70b", "abliterate")
            assert feasible is False
            assert "Estimated memory" in reason
            assert "exceeds" in reason

    def test_custom_max_memory_pct(self):
        """Lower threshold should reject more models."""
        with patch(
            "adversarial_framework.services.finetuning.memory.get_available_memory_gb",
            return_value=10.0,
        ):
            # 8B abliterate = 6.5GB, 50% of 10GB = 5GB → infeasible
            feasible, _ = check_memory_feasibility("llama3:8b", "abliterate", max_memory_pct=0.50)
            assert feasible is False
