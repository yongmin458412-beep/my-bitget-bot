"""Structured output schemas for the OpenAI Responses API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


def _make_schema_strict(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively enforce the strict JSON Schema shape expected by Responses API."""

    if not isinstance(schema, dict):
        return schema

    properties = schema.get("properties")
    if isinstance(properties, dict):
        schema["required"] = list(properties.keys())
        schema["additionalProperties"] = False
        for value in properties.values():
            _make_schema_strict(value)

    items = schema.get("items")
    if isinstance(items, dict):
        _make_schema_strict(items)

    for key in ("anyOf", "allOf", "oneOf"):
        variants = schema.get(key)
        if isinstance(variants, list):
            for variant in variants:
                if isinstance(variant, dict):
                    _make_schema_strict(variant)

    defs = schema.get("$defs")
    if isinstance(defs, dict):
        for value in defs.values():
            if isinstance(value, dict):
                _make_schema_strict(value)

    return schema


class StrictSchemaModel(BaseModel):
    """Base model that emits Responses API friendly strict JSON Schema."""

    @classmethod
    def json_schema_strict(cls) -> dict[str, Any]:
        """Return strict JSON Schema for Responses API."""

        return _make_schema_strict(cls.model_json_schema())


class NewsAnalysis(StrictSchemaModel):
    """Structured news analysis schema."""

    summary_ko: str = Field(description="기사/이벤트를 2~3문장으로 요약한 한국어 문장")
    impacted_assets: list[str] = Field(default_factory=list, description="영향받는 자산 티커")
    impact_level: Literal["low", "medium", "high"]
    direction_bias: Literal["bullish", "bearish", "neutral", "uncertain"]
    validity_window_minutes: int
    confidence: float
    event_type: str
    should_block_new_entries: bool
    notes: str


class WhyExplanation(StrictSchemaModel):
    """Human-readable strategy explanation."""

    symbol: str
    summary_ko: str
    bullet_points: list[str]
    regime_commentary: str
    risk_commentary: str


class JournalSummary(StrictSchemaModel):
    """Trade journal summary schema."""

    summary_ko: str
    wins: int
    losses: int
    key_lessons: list[str]
    next_focus: list[str]

