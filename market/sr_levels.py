"""Support and resistance level construction."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .indicators import equal_highs_lows, swing_points, vwap
from .sessions import asia_session_high_low, london_session_high_low, previous_day_high_low


@dataclass(slots=True)
class LevelSet:
    """Collection of structural levels."""

    levels: dict[str, float] = field(default_factory=dict)
    tags: dict[str, list[float]] = field(default_factory=dict)


class SupportResistanceDetector:
    """Build the level map used by both strategies."""

    def build_levels(self, df_3m: pd.DataFrame, df_15m: pd.DataFrame) -> LevelSet:
        """Compute structural levels from multi-timeframe data."""

        levels: dict[str, float] = {}
        tags: dict[str, list[float]] = {}

        prev_high, prev_low = previous_day_high_low(df_15m)
        if prev_high is not None and prev_low is not None:
            levels["prev_day_high"] = prev_high
            levels["prev_day_low"] = prev_low

        asia = asia_session_high_low(df_15m)
        if asia:
            levels["asia_high"] = asia.high
            levels["asia_low"] = asia.low

        london = london_session_high_low(df_15m)
        if london:
            levels["london_high"] = london.high
            levels["london_low"] = london.low

        recent_4h = df_15m.tail(16)
        if not recent_4h.empty:
            swings = swing_points(recent_4h, window=2)
            if swings["highs"]:
                levels["swing_high_4h"] = swings["highs"][-1][1]
            if swings["lows"]:
                levels["swing_low_4h"] = swings["lows"][-1][1]
            levels["range_high_recent"] = float(recent_4h["high"].max())
            levels["range_low_recent"] = float(recent_4h["low"].min())

        if not df_3m.empty:
            levels["vwap"] = float(vwap(df_3m).iloc[-1])
            eq = equal_highs_lows(df_3m, lookback=50, tolerance_bps=6)
            tags.update(eq)

        return LevelSet(levels=levels, tags=tags)

