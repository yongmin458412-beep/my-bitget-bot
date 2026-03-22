"""Helpers for deriving structural levels from existing market modules."""

from __future__ import annotations

from typing import Any

import pandas as pd

from market.indicators import equal_highs_lows, swing_points
from market.sessions import asia_session_high_low, london_session_high_low, previous_day_high_low


def find_recent_swings(df: pd.DataFrame, lookback: int = 60, side: str | None = None, window: int = 2) -> dict[str, list[float]]:
    """Return recent swing highs and lows from the tail of a frame."""

    recent = df.tail(lookback)
    swings = swing_points(recent, window=window)
    highs = [float(price) for _, price in swings["highs"]]
    lows = [float(price) for _, price in swings["lows"]]
    payload = {
        "swing_highs": highs,
        "swing_lows": lows,
    }
    if side == "long":
        return {"swing_lows": lows}
    if side == "short":
        return {"swing_highs": highs}
    return payload


def detect_session_levels(df: pd.DataFrame, timezone: str = "UTC") -> dict[str, float]:
    """Return session levels using the repo's existing UTC helpers."""

    del timezone  # current session helpers already operate on UTC-indexed frames
    levels: dict[str, float] = {}
    prev_high, prev_low = previous_day_high_low(df)
    if prev_high is not None and prev_low is not None:
        levels["prev_day_high"] = prev_high
        levels["prev_day_low"] = prev_low
    asia = asia_session_high_low(df)
    if asia:
        levels["asia_high"] = asia.high
        levels["asia_low"] = asia.low
    london = london_session_high_low(df)
    if london:
        levels["london_high"] = london.high
        levels["london_low"] = london.low
    return levels


def detect_equal_highs_lows(df: pd.DataFrame, threshold: float = 6.0) -> dict[str, list[float]]:
    """Return equal highs/lows using a bps threshold."""

    return equal_highs_lows(df, lookback=50, tolerance_bps=threshold)


def detect_range_boundaries(df: pd.DataFrame, lookback: int = 16) -> dict[str, float]:
    """Return a simple recent range envelope."""

    recent = df.tail(lookback)
    if recent.empty:
        return {}
    range_high = float(recent["high"].max())
    range_low = float(recent["low"].min())
    return {
        "range_high": range_high,
        "range_low": range_low,
        "range_mid": (range_high + range_low) / 2.0,
        "range_width": max(range_high - range_low, 0.0),
    }


def build_structure_snapshot(
    *,
    df_entry: pd.DataFrame,
    df_structure: pd.DataFrame,
    range_lookback: int = 16,
    swing_lookback: int = 60,
    equal_threshold: float = 6.0,
) -> dict[str, Any]:
    """Build a merged structural snapshot for stop/target logic."""

    session_levels = detect_session_levels(df_structure)
    equal_levels = detect_equal_highs_lows(df_entry, threshold=equal_threshold)
    recent_swings_entry = find_recent_swings(df_entry, lookback=swing_lookback)
    recent_swings_structure = find_recent_swings(df_structure, lookback=min(max(range_lookback * 4, 20), len(df_structure)))
    range_boundaries = detect_range_boundaries(df_structure, lookback=range_lookback)
    return {
        "session_levels": session_levels,
        "equal_levels": equal_levels,
        "recent_swings_entry": recent_swings_entry,
        "recent_swings_structure": recent_swings_structure,
        "range_boundaries": range_boundaries,
    }
