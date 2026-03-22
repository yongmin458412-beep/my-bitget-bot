"""Rationale payload tests for Vincent and Liquidity strategies."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from core.enums import RegimeType, Side, StrategyName
from strategy.base import SignalContext
from strategy.break_retest import BreakRetestStrategy
from strategy.liquidity_raid import LiquidityRaidStrategy


def _break_context() -> SignalContext:
    index_3m = pd.date_range("2026-03-18 00:00:00", periods=70, freq="3min", tz="UTC")
    close = np.linspace(99.2, 100.4, len(index_3m))
    close[-3:] = [100.05, 99.98, 100.35]
    open_ = close - 0.08
    open_[-1] = close[-1] - 0.20
    high = np.maximum(open_, close) + 0.12
    low = np.minimum(open_, close) - 0.05
    low[-1] = 99.92
    volume = np.linspace(1000, 1300, len(index_3m))
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
    index_15m = pd.date_range("2026-03-17 00:00:00", periods=48, freq="15min", tz="UTC")
    ladder = np.linspace(98.5, 101.0, len(index_15m))
    frame_15m = pd.DataFrame(
        {
            "open": ladder,
            "high": ladder + 0.3,
            "low": ladder - 0.3,
            "close": ladder + 0.08,
            "volume": np.linspace(1700, 2500, len(index_15m)),
            "quote_volume": np.linspace(180000, 300000, len(index_15m)),
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
        levels={"prev_day_high": 100.0, "asia_high": 99.8, "london_high": 100.2, "vwap": 99.9},
        level_tags={"equal_highs": [], "equal_lows": []},
        regime=RegimeType.TRENDING,
        regime_notes=[],
        ticker={"last_price": 100.35, "spread_bps": 2.0},
        orderbook={"spread_bps": 2.0},
        historical_stats={"win_rate": 0.5, "avg_win_r": 1.9, "avg_loss_r": 1.0},
    )


def _liquidity_context() -> SignalContext:
    index_3m = pd.date_range("2026-03-18 00:00:00", periods=80, freq="3min", tz="UTC")
    close = np.linspace(101.8, 100.5, len(index_3m))
    close[-4:] = [100.4, 99.88, 100.12, 100.35]
    open_ = close + 0.05
    open_[-1] = close[-1] - 0.18
    high = np.maximum(open_, close) + 0.08
    low = np.minimum(open_, close) - 0.08
    low[-2] = 99.72
    volume = np.linspace(900, 1200, len(index_3m))
    volume[-1] = 2100
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
    index_15m = pd.date_range("2026-03-17 00:00:00", periods=56, freq="15min", tz="UTC")
    ladder = np.linspace(103.0, 99.5, len(index_15m))
    frame_15m = pd.DataFrame(
        {
            "open": ladder,
            "high": ladder + 0.35,
            "low": ladder - 0.35,
            "close": ladder - 0.05,
            "volume": np.linspace(1600, 2600, len(index_15m)),
            "quote_volume": np.linspace(170000, 290000, len(index_15m)),
        },
        index=index_15m,
    )
    frame_1h = frame_15m.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "quote_volume": "sum"}
    ).dropna()
    return SignalContext(
        symbol="XRPUSDT",
        product_type="USDT-FUTURES",
        frames={"3m": frame_3m, "5m": frame_5m, "15m": frame_15m, "1H": frame_1h},
        levels={"asia_low": 99.9, "london_low": 100.1, "prev_day_low": 99.85},
        level_tags={"equal_highs": [], "equal_lows": [99.9]},
        regime=RegimeType.RANGING,
        regime_notes=[],
        ticker={"last_price": 100.35, "spread_bps": 2.0},
        orderbook={"spread_bps": 2.0},
        historical_stats={"win_rate": 0.47, "avg_win_r": 1.7, "avg_loss_r": 1.0},
    )


def test_break_retest_signal_contains_entry_reason_payload() -> None:
    """Vincent strategy should populate structured Korean entry reasons for charting."""

    strategy = BreakRetestStrategy()
    targets = [
        {"price": 100.9, "reason": "previous_high", "priority": 1, "inside_range_middle": False},
        {"price": 101.5, "reason": "session_high", "priority": 2, "inside_range_middle": False},
    ]
    with (
        patch("strategy.break_retest.atr", side_effect=lambda df: pd.Series([2.0] * len(df), index=df.index)),
        patch("strategy.break_retest._range_middle", return_value=False),
        patch("strategy.break_retest.detect_structural_stop", return_value=(99.6, "retest_low_atr_buffer", {})),
        patch("strategy.break_retest.detect_structural_targets", return_value=targets),
        patch("strategy.break_retest.bullish_confirmation", return_value=True),
        patch("strategy.break_retest.volume_recovered", return_value=True),
    ):
        signal = strategy.evaluate(_break_context())

    assert signal is not None
    assert signal.strategy == StrategyName.BREAK_RETEST
    assert signal.side == Side.LONG
    assert signal.rationale["entry_reason_title"]
    assert len(signal.rationale["entry_reason_lines"]) >= 3
    assert signal.rationale["chart_levels"]
    assert signal.rationale["chart_zones"]
    assert signal.rationale["chart_marker"]["label"] == "확인 캔들"


def test_liquidity_raid_signal_contains_entry_reason_payload() -> None:
    """Liquidity reclaim strategy should populate reclaim-focused structured reasons."""

    strategy = LiquidityRaidStrategy()
    targets = [
        {"price": 100.7, "reason": "previous_high", "priority": 1, "inside_range_middle": False},
        {"price": 101.2, "reason": "session_high", "priority": 2, "inside_range_middle": False},
    ]
    with (
        patch("strategy.liquidity_raid.detect_structural_stop", return_value=(99.5, "recent_swing_low", {})),
        patch("strategy.liquidity_raid.detect_structural_targets", return_value=targets),
        patch("strategy.liquidity_raid.bullish_confirmation", return_value=True),
        patch("strategy.liquidity_raid.mss_bullish", return_value=True),
        patch("strategy.liquidity_raid.volume_recovered", return_value=True),
    ):
        signal = strategy.evaluate(_liquidity_context())

    assert signal is not None
    assert signal.strategy == StrategyName.LIQUIDITY_RAID
    assert signal.side == Side.LONG
    assert signal.rationale["entry_reason_title"]
    assert len(signal.rationale["entry_reason_lines"]) >= 3
    assert signal.rationale["chart_levels"]
    assert signal.rationale["chart_zones"]
    assert signal.rationale["chart_marker"]["label"] == "Reclaim"
