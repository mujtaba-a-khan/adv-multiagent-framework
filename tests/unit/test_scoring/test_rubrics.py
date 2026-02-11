"""Tests for scoring rubrics."""

from __future__ import annotations

from adversarial_framework.scoring.rubrics import (
    ALL_RUBRICS,
    COHERENCE_RUBRIC,
    SEVERITY_RUBRIC,
    SPECIFICITY_RUBRIC,
    ScoreLevel,
    format_all_rubrics_for_prompt,
    format_rubric_for_prompt,
)


class TestScoreLevel:
    def test_creation(self) -> None:
        level = ScoreLevel(score=5, label="Moderate", description="Mid-level")
        assert level.score == 5
        assert level.label == "Moderate"
        assert level.description == "Mid-level"

    def test_frozen(self) -> None:
        level = ScoreLevel(score=1, label="Low", description="Low level")
        try:
            level.score = 2  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except AttributeError:
            pass


class TestScoringRubric:
    def test_severity_rubric_structure(self) -> None:
        assert SEVERITY_RUBRIC.name == "severity"
        assert SEVERITY_RUBRIC.min_score == 1
        assert SEVERITY_RUBRIC.max_score == 10
        assert len(SEVERITY_RUBRIC.levels) == 10

    def test_specificity_rubric_structure(self) -> None:
        assert SPECIFICITY_RUBRIC.name == "specificity"
        assert SPECIFICITY_RUBRIC.min_score == 1
        assert SPECIFICITY_RUBRIC.max_score == 10
        assert len(SPECIFICITY_RUBRIC.levels) == 10

    def test_coherence_rubric_structure(self) -> None:
        assert COHERENCE_RUBRIC.name == "coherence"
        assert COHERENCE_RUBRIC.min_score == 1
        assert COHERENCE_RUBRIC.max_score == 10
        assert len(COHERENCE_RUBRIC.levels) == 10

    def test_get_level_found(self) -> None:
        level = SEVERITY_RUBRIC.get_level(1)
        assert level is not None
        assert level.label == "Benign"

    def test_get_level_max(self) -> None:
        level = SEVERITY_RUBRIC.get_level(10)
        assert level is not None
        assert level.label == "Catastrophic"

    def test_get_level_not_found(self) -> None:
        assert SEVERITY_RUBRIC.get_level(0) is None
        assert SEVERITY_RUBRIC.get_level(11) is None

    def test_all_rubrics_dict(self) -> None:
        assert set(ALL_RUBRICS.keys()) == {"severity", "specificity", "coherence"}
        assert ALL_RUBRICS["severity"] is SEVERITY_RUBRIC
        assert ALL_RUBRICS["specificity"] is SPECIFICITY_RUBRIC
        assert ALL_RUBRICS["coherence"] is COHERENCE_RUBRIC

    def test_levels_cover_full_range(self) -> None:
        for rubric in ALL_RUBRICS.values():
            scores = [level.score for level in rubric.levels]
            assert scores == list(range(1, 11))


class TestFormatRubric:
    def test_format_rubric_includes_name(self) -> None:
        text = format_rubric_for_prompt(SEVERITY_RUBRIC)
        assert "severity" in text
        assert "Harm Severity" in text

    def test_format_rubric_includes_all_levels(self) -> None:
        text = format_rubric_for_prompt(SEVERITY_RUBRIC)
        for level in SEVERITY_RUBRIC.levels:
            assert level.label in text
            assert str(level.score) in text

    def test_format_rubric_includes_score_range(self) -> None:
        text = format_rubric_for_prompt(SEVERITY_RUBRIC)
        assert "1-10" in text

    def test_format_all_rubrics(self) -> None:
        text = format_all_rubrics_for_prompt()
        assert "severity" in text
        assert "specificity" in text
        assert "coherence" in text
        # All three rubrics should be separated
        assert text.count("Score range:") == 3
