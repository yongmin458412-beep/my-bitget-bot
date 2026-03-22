"""Structural stop selection."""

from __future__ import annotations

from math import isfinite
from typing import Any

from core.enums import Side
from risk.stops import apply_atr_buffer
from strategy.base import SignalContext

from .structure_levels import build_structure_snapshot


def _is_valid_stop(*, entry_price: float, stop_price: float, side: Side) -> bool:
    """Return True if the stop is on the correct side of entry."""

    if side == Side.LONG:
        return stop_price < entry_price
    return stop_price > entry_price


def _normalized_risk_pct(entry_price: float, stop_price: float) -> float:
    """Return stop distance as a percentage of entry."""

    return abs(entry_price - stop_price) / max(abs(entry_price), 1e-9)


def detect_structural_stop(
    signal_context: SignalContext,
    setup_context: dict[str, Any],
    *,
    use_atr_buffer: bool = True,
    atr_buffer_multiplier: float = 0.15,
    min_stop_distance_pct: float = 0.005,
    max_stop_distance_pct: float = 0.04,
) -> tuple[float | None, str | None, dict[str, Any]]:
    """Pick the best structural invalidation level for the setup."""

    side: Side = setup_context["side"]
    entry_price = float(setup_context["entry_price"])
    atr_value = float(setup_context.get("atr") or 0.0)
    structure = build_structure_snapshot(
        df_entry=signal_context.frames["3m"],
        df_structure=signal_context.frames["15m"],
    )
    range_boundaries = setup_context.get("range_boundaries") or structure["range_boundaries"]
    recent_entry_swings = setup_context.get("recent_swings_entry") or structure["recent_swings_entry"]
    recent_structure_swings = setup_context.get("recent_swings_structure") or structure["recent_swings_structure"]
    session_levels = structure["session_levels"]

    if side == Side.LONG:
        # 👑 전일저점을 최우선으로 (전고점 SL 구조)
        candidates = [
            ("previous_day_low", session_levels.get("prev_day_low")),
            ("retest_low", setup_context.get("retest_low")),
            ("recent_swing_low", next((price for price in reversed(recent_entry_swings.get("swing_lows", [])) if price < entry_price), None)),
            ("liquidity_sweep_low", setup_context.get("sweep_low")),
            ("range_low", range_boundaries.get("range_low")),
            ("higher_timeframe_swing_low", next((price for price in reversed(recent_structure_swings.get("swing_lows", [])) if price < entry_price), None)),
        ]
    else:
        # 👑 전일고점을 최우선으로 (전저점 SL 구조)
        candidates = [
            ("previous_day_high", session_levels.get("prev_day_high")),
            ("retest_high", setup_context.get("retest_high")),
            ("recent_swing_high", next((price for price in reversed(recent_entry_swings.get("swing_highs", [])) if price > entry_price), None)),
            ("liquidity_sweep_high", setup_context.get("sweep_high")),
            ("range_high", range_boundaries.get("range_high")),
            ("higher_timeframe_swing_high", next((price for price in reversed(recent_structure_swings.get("swing_highs", [])) if price > entry_price), None)),
        ]

    rejected: list[dict[str, Any]] = []
    for reason, raw_level in candidates:
        if raw_level is None:
            continue
        raw_stop = float(raw_level)
        if not isfinite(raw_stop):
            continue
        stop_price = raw_stop
        if use_atr_buffer and atr_value > 0:
            stop_price = apply_atr_buffer(stop_price, atr_value, side, multiplier=atr_buffer_multiplier)
        if not _is_valid_stop(entry_price=entry_price, stop_price=stop_price, side=side):
            rejected.append({"reason": reason, "raw_stop": raw_stop, "rejected": "wrong_side"})
            continue
        risk_pct = _normalized_risk_pct(entry_price, stop_price)
        if risk_pct < min_stop_distance_pct:
            rejected.append({"reason": reason, "raw_stop": raw_stop, "rejected": "too_close", "risk_pct": risk_pct})
            continue
        if risk_pct > max_stop_distance_pct:
            rejected.append({"reason": reason, "raw_stop": raw_stop, "rejected": "too_far", "risk_pct": risk_pct})
            continue
        return (
            stop_price,
            f"{reason}_atr_buffer" if use_atr_buffer and atr_value > 0 else reason,
            {
                "raw_stop": raw_stop,
                "atr": atr_value,
                "risk_pct": risk_pct,
                "candidate_reason": reason,
                "rejected_candidates": rejected,
            },
        )

    # ── 합성 SL fallback: 모든 구조적 SL 실패 시 ATR 기반 SL 생성 ──
    if atr_value > 0:
        if side == Side.LONG:
            synth_stop = entry_price - atr_value * 2.0
        else:
            synth_stop = entry_price + atr_value * 2.0
        if use_atr_buffer:
            synth_stop = apply_atr_buffer(synth_stop, atr_value, side, multiplier=atr_buffer_multiplier)
        risk_pct = _normalized_risk_pct(entry_price, synth_stop)
        if _is_valid_stop(entry_price=entry_price, stop_price=synth_stop, side=side) and risk_pct <= max_stop_distance_pct:
            return (
                synth_stop,
                "synthetic_atr_fallback",
                {
                    "raw_stop": synth_stop,
                    "atr": atr_value,
                    "risk_pct": risk_pct,
                    "candidate_reason": "synthetic_atr_1.5x",
                    "rejected_candidates": rejected,
                },
            )

    return None, None, {"rejected_candidates": rejected}
