"""Tests for the multi-dimensional scorer."""

from __future__ import annotations

import pytest

from adversarial_framework.agents.analyzer.scorer import run_scorer, _parse_scorer_response


class TestParseScorer:
    def test_valid_json(self):
        result = _parse_scorer_response(
            '{"severity": 7, "specificity": 5, "coherence": 8}'
        )
        assert result["severity"] == 7.0
        assert result["specificity"] == 5.0
        assert result["coherence"] == 8.0

    def test_clamps_values(self):
        result = _parse_scorer_response(
            '{"severity": 15, "specificity": -2, "coherence": 0}'
        )
        assert result["severity"] == 10.0
        assert result["specificity"] == 1.0
        assert result["coherence"] == 1.0

    def test_markdown_wrapped(self):
        text = '```json\n{"severity": 6, "specificity": 4, "coherence": 7}\n```'
        result = _parse_scorer_response(text)
        assert result["severity"] == 6.0

    def test_invalid_json_defaults(self):
        result = _parse_scorer_response("not valid json at all")
        assert result["severity"] == 5.0
        assert result["specificity"] == 5.0
        assert result["coherence"] == 5.0

    def test_missing_fields_default(self):
        result = _parse_scorer_response('{"severity": 3}')
        assert result["severity"] == 3.0
        assert result["specificity"] == 5.0  # default
        assert result["coherence"] == 5.0  # default


class TestRunScorer:
    @pytest.mark.asyncio
    async def test_run_scorer(self, mock_provider_factory):
        provider = mock_provider_factory(
            ['{"severity": 8, "specificity": 6, "coherence": 7}']
        )
        result = await run_scorer(
            provider=provider,
            model="mock",
            target_response="Some harmful content here",
        )
        assert result["severity_score"] == 8.0
        assert result["specificity_score"] == 6.0
        assert result["coherence_score"] == 7.0
