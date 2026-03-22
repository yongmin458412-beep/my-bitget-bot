"""Momentum pullback strategy tests."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from core.enums import RegimeType, Side, StrategyName
from strategy.base import SignalContext
from strategy.momentum_pullback import MomentumPullbackStrategy


def _context() -> SignalContext:
    index_3m = pd.date_range("2026-03-18 00:00:00", periods=100, freq="3min", tz="UTC")
    close = np.linspace(100.0, 109.0, len(index_3m))
    close[-12:] = [106.0, 106.8, 107.7, 108.6, 109.4, 110.1, 109.9, 109.6, 109.7, 109.9, 110.0, 110.4]
    open_ = close - 0.12
    open_[-1] = close[-1] - 0.22
    high = np.maximum(open_, close) + 0.15
    low = np.minimum(open_, close) - 0.08
    high[-1] = close[-1] + 0.05
    volume = np.linspace(900, 1500, len(index_3m))
    volume[-1] = 2200
    frame_3m = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
        },
        index=index_3m,
    )
    frame_5m = frame_3m.resample("5min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "quote_volume": "sum"}
    ).dropna()
    index_15m = pd.date_range("2026-03-17 00:00:00", periods=80, freq="15min", tz="UTC")
    ladder = np.linspace(98.0, 112.0, len(index_15m))
    frame_15m = pd.DataFrame(
        {
            "open": ladder,
            "high": ladder + 0.35,
            "low": ladder - 0.35,
            "close": ladder + 0.1,
            "volume": np.linspace(2000, 3500, len(index_15m)),
            "quote_volume": np.linspace(250000, 420000, len(index_15m)),
        },
        index=index_15m,
    )
    frame_1h = frame_15m.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "quote_volume": "sum"}
    ).dropna()
    return SignalContext(
        symbol="ETHUSDT",
        product_type="USDT-FUTURES",
        frames={"3m": frame_3m, "5m": frame_5m, "15m": frame_15m, "1H": frame_1h},
        levels={"prev_day_high": 112.0, "london_high": 111.5},
        level_tags={"equal_highs": [], "equal_lows": []},
        regime=RegimeType.TRENDING,
        regime_notes=[],
        ticker={"last_price": 110.4, "spread_bps": 2.0},
        orderbook={"spread_bps": 2.0},
        historical_stats={"win_rate": 0.5, "avg_win_r": 1.9, "avg_loss_r": 1.0},
    )


def test_momentum_pullback_emits_continuation_signal() -> None:
    """Strong impulse followed by shallow pullback should create a continuation signal."""

    strategy = MomentumPullbackStrategy()
    targets = [
        {"price": 111.2, "reason": "recent_swing_high", "priority": 1, "inside_range_middle": False},
        {"price": 112.0, "reason": "previous_high", "priority": 2, "inside_range_middle": False},
    ]
    with (
        patch("strategy.momentum_pullback.detect_structural_stop", return_value=(109.4, "retest_low_atr_buffer", {})),
        patch("strategy.momentum_pullback.detect_structural_targets", return_value=targets),
    ):
        signal = strategy.evaluate(_context())

    assert signal is not None
    assert signal.strategy == StrategyName.MOMENTUM_PULLBACK
    assert signal.side == Side.LONG
    assert signal.tp2_price == 112.0
    assert signal.rationale["entry_reason_title"]
    assert len(signal.rationale["entry_reason_lines"]) >= 3
    assert signal.rationale["chart_levels"]
    assert signal.rationale["chart_zones"]
    assert signal.rationale["chart_marker"]["label"] == "재개 캔들"
