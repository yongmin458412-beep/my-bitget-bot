"""Structural stop selection tests."""

from __future__ import annotations

import pandas as pd

from core.enums import RegimeType, Side
from strategy.base import SignalContext
from strategy.structural_stops import detect_structural_stop


def _frame() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=80, freq="15min", tz="UTC")
    base = pd.Series(range(80), index=index, dtype=float)
    return pd.DataFrame(
        {
            "open": 100 + base * 0.1,
            "high": 100.4 + base * 0.1,
            "low": 99.6 + base * 0.1,
            "close": 100.1 + base * 0.1,
            "volume": 1000 + base * 10,
            "quote_volume": 100000 + base * 100,
        },
        index=index,
    )


def _context() -> SignalContext:
    frame_3m = _frame().resample("3min").ffill().dropna()
    frame_15m = _frame()
    return SignalContext(
        symbol="BTCUSDT",
        product_type="USDT-FUTURES",
        frames={"3m": frame_3m, "5m": frame_3m.tail(80), "15m": frame_15m, "1H": frame_15m.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "quote_volume": "sum"}).dropna()},
        levels={"prev_day_low": 99.0, "prev_day_high": 110.0},
        level_tags={},
        regime=RegimeType.TREND,
        regime_notes=[],
        ticker={"last_price": 105.0, "spread_bps": 2.0},
        orderbook={"spread_bps": 2.0},
        historical_stats={"win_rate": 0.5, "avg_win_r": 1.8, "avg_loss_r": 1.0},
    )


def test_detect_structural_stop_prefers_retest_low_for_long() -> None:
    """Long break-retest should choose the retest low before deeper structure."""

    context = _context()
    stop_price, stop_reason, _ = detect_structural_stop(
        context,
        {
            "side": Side.LONG,
            "entry_price": 105.0,
            "atr": 1.0,
            "retest_low": 103.9,
            "range_boundaries": {"range_low": 101.0, "range_high": 109.0},
            "recent_swings_entry": {"swing_lows": [102.5, 103.2], "swing_highs": [106.5]},
            "recent_swings_structure": {"swing_lows": [101.8], "swing_highs": [108.2]},
        },
    )
    assert stop_price is not None
    assert stop_reason == "retest_low_atr_buffer"


def test_detect_structural_stop_prefers_retest_high_for_short() -> None:
    """Short break-retest should choose the retest high before deeper structure."""

    context = _context()
    stop_price, stop_reason, _ = detect_structural_stop(
        context,
        {
            "side": Side.SHORT,
            "entry_price": 105.0,
            "atr": 1.0,
            "retest_high": 106.1,
            "range_boundaries": {"range_low": 101.0, "range_high": 109.0},
            "recent_swings_entry": {"swing_lows": [102.5], "swing_highs": [106.8, 107.2]},
            "recent_swings_structure": {"swing_lows": [101.8], "swing_highs": [108.2]},
        },
    )
    assert stop_price is not None
    assert stop_reason == "retest_high_atr_buffer"
