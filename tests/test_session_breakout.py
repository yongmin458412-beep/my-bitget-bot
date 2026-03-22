"""Session breakout strategy tests."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from core.enums import RegimeType, Side, StrategyName
from strategy.base import SignalContext
from strategy.session_breakout import SessionBreakoutStrategy


def _context() -> SignalContext:
    index_3m = pd.date_range("2026-03-18 09:00:00", periods=90, freq="3min", tz="UTC")
    base = np.linspace(100.0, 101.6, len(index_3m))
    close = base.copy()
    close[-3:] = [101.8, 101.95, 102.35]
    open_ = close - 0.08
    open_[-1] = close[-1] - 0.24
    high = close + 0.12
    low = open_ - 0.03
    high[-1] = close[-1] + 0.05
    volume = np.linspace(1000, 1400, len(index_3m))
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
    index_15m = pd.date_range("2026-03-17 00:00:00", periods=64, freq="15min", tz="UTC")
    ladder = np.linspace(98.0, 101.2, len(index_15m))
    frame_15m = pd.DataFrame(
        {
            "open": ladder,
            "high": ladder + 0.4,
            "low": ladder - 0.4,
            "close": ladder + 0.1,
            "volume": np.linspace(2000, 3200, len(index_15m)),
            "quote_volume": np.linspace(220000, 350000, len(index_15m)),
        },
        index=index_15m,
    )
    frame_1h = frame_15m.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "quote_volume": "sum"}
    ).dropna()
    return SignalContext(
        symbol="BTCUSDT",
        product_type="USDT-FUTURES",
        frames={"3m": frame_3m, "5m": frame_5m, "15m": frame_15m, "1H": frame_1h},
        levels={
            "london_high": 102.0,
            "london_low": 100.7,
            "asia_high": 101.4,
            "asia_low": 99.2,
        },
        level_tags={"equal_highs": [], "equal_lows": []},
        regime=RegimeType.EXPANSION,
        regime_notes=[],
        ticker={"last_price": 102.35, "spread_bps": 2.0},
        orderbook={"spread_bps": 2.0},
        historical_stats={"win_rate": 0.48, "avg_win_r": 1.8, "avg_loss_r": 1.0},
    )


def test_session_breakout_produces_long_signal() -> None:
    """Compressed session breakout should emit a deterministic long signal."""

    strategy = SessionBreakoutStrategy()
    targets = [
        {"price": 103.0, "reason": "previous_high", "priority": 1, "inside_range_middle": False},
        {"price": 103.8, "reason": "session_high", "priority": 2, "inside_range_middle": False},
    ]
    with (
        patch("strategy.session_breakout.detect_structural_stop", return_value=(101.4, "range_low_atr_buffer", {})),
        patch("strategy.session_breakout.detect_structural_targets", return_value=targets),
    ):
        signal = strategy.evaluate(_context())

    assert signal is not None
    assert signal.strategy == StrategyName.SESSION_BREAKOUT
    assert signal.side == Side.LONG
    assert signal.tp1_price == 103.0
    assert signal.rationale["entry_reason_title"]
    assert len(signal.rationale["entry_reason_lines"]) >= 3
    assert signal.rationale["chart_levels"]
    assert signal.rationale["chart_zones"]
    assert signal.rationale["chart_marker"]["label"] == "돌파 캔들"
