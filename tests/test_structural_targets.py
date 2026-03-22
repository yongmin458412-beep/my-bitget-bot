"""Structural target generation tests."""

from __future__ import annotations

import pandas as pd

from core.enums import RegimeType, Side
from strategy.base import SignalContext
from strategy.structural_targets import detect_structural_targets, merge_close_targets


def _frame() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=80, freq="15min", tz="UTC")
    values = pd.Series(range(80), index=index, dtype=float)
    return pd.DataFrame(
        {
            "open": 100 + values * 0.1,
            "high": 100.5 + values * 0.12,
            "low": 99.5 + values * 0.08,
            "close": 100.2 + values * 0.1,
            "volume": 1000 + values * 10,
            "quote_volume": 100000 + values * 100,
        },
        index=index,
    )


def _context() -> SignalContext:
    frame_15m = _frame()
    frame_3m = frame_15m.resample("3min").ffill().dropna()
    frame_1h = frame_15m.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "quote_volume": "sum"}).dropna()
    return SignalContext(
        symbol="ETHUSDT",
        product_type="USDT-FUTURES",
        frames={"3m": frame_3m, "5m": frame_3m.tail(120), "15m": frame_15m, "1H": frame_1h},
        levels={
            "prev_day_high": 109.5,
            "prev_day_low": 98.0,
            "asia_high": 108.3,
            "asia_low": 99.1,
            "london_high": 110.2,
            "london_low": 98.9,
            "range_high_recent": 108.8,
            "range_low_recent": 100.2,
        },
        level_tags={"equal_highs": [108.25, 108.28], "equal_lows": [99.2]},
        regime=RegimeType.RANGE,
        regime_notes=[],
        ticker={"last_price": 104.0, "spread_bps": 3.0},
        orderbook={"spread_bps": 3.0},
        historical_stats={"win_rate": 0.5, "avg_win_r": 1.8, "avg_loss_r": 1.0},
    )


def test_detect_structural_targets_builds_tp1_tp2_for_long() -> None:
    """Liquidity-style long setup should produce ordered structural targets."""

    targets = detect_structural_targets(
        _context(),
        {
            "side": Side.LONG,
            "entry_price": 104.0,
            "atr": 0.8,
        },
    )
    assert len(targets) >= 2
    assert targets[0]["price"] < targets[1]["price"]


def test_merge_close_targets_collapses_nearby_levels() -> None:
    """Nearby structural levels should be merged into a single target zone."""

    merged = merge_close_targets(
        [
            {"price": 108.2500, "reason": "equal_highs", "reasons": ["equal_highs"], "priority": 1, "inside_range_middle": False},
            {"price": 108.2800, "reason": "previous_high", "reasons": ["previous_high"], "priority": 2, "inside_range_middle": False},
            {"price": 110.0000, "reason": "session_high", "reasons": ["session_high"], "priority": 3, "inside_range_middle": False},
        ],
        threshold_pct=0.001,
    )
    assert len(merged) == 2
