"""Tests for strict OpenAI response schemas."""

from __future__ import annotations

from ai.schemas import JournalSummary, NewsAnalysis, WhyExplanation


def test_news_analysis_schema_marks_all_properties_required() -> None:
    """Strict schema should require every property for Responses API."""

    schema = NewsAnalysis.json_schema_strict()
    assert sorted(schema["required"]) == sorted(schema["properties"].keys())
    assert "impacted_assets" in schema["required"]
    assert schema["additionalProperties"] is False


def test_other_schemas_are_also_strict() -> None:
    """Other structured outputs should follow the same strict rule."""

    for model in (WhyExplanation, JournalSummary):
        schema = model.json_schema_strict()
        assert sorted(schema["required"]) == sorted(schema["properties"].keys())
        assert schema["additionalProperties"] is False
