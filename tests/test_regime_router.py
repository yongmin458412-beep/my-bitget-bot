"""Regime router tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.enums import RegimeType, StrategyName
from market.market_regime import MarketRegimeClassifier


def _frame(
    *,
    periods: int,
    start: float,
    drift: float,
    volume_start: float,
    volume_end: float,
    wiggle_scale: float | None = None,
    wick_size: float | None = None,
) -> pd.DataFrame:
    """Build a synthetic OHLCV frame for regime tests."""

    index = pd.date_range("2026-03-18 00:00:00", periods=periods, freq="15min", tz="UTC")
    base = start + np.cumsum(np.full(periods, drift))
    wiggle_base = max(abs(drift) * 4, 0.4) if wiggle_scale is None else wiggle_scale
    wiggle = np.sin(np.linspace(0, 8, periods)) * wiggle_base
    close = base + wiggle
    open_ = close - drift * 0.5
    wick = max(abs(drift) * 2, 0.05) if wick_size is None else wick_size
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    volume = np.linspace(volume_start, volume_end, periods)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def test_regime_classifier_enables_trend_strategies() -> None:
    """Trending regime should prefer Vincent and Momentum Pullback."""

    classifier = MarketRegimeClassifier()
    df_15m = _frame(periods=160, start=100.0, drift=0.9, volume_start=1000, volume_end=2000)
    df_1h = _frame(periods=160, start=100.0, drift=2.4, volume_start=1200, volume_end=2400)

    snapshot = classifier.classify(df_15m, df_1h)

    assert snapshot.regime in {RegimeType.TRENDING, RegimeType.EXPANSION}
    if snapshot.regime == RegimeType.TRENDING:
        assert StrategyName.BREAK_RETEST in snapshot.allowed_strategies
        assert StrategyName.MOMENTUM_PULLBACK in snapshot.allowed_strategies


def test_regime_classifier_identifies_dead_market() -> None:
    """Extremely quiet conditions should disable entries."""

    classifier = MarketRegimeClassifier()
    df_15m = _frame(periods=160, start=100.0, drift=0.0, volume_start=50, volume_end=70, wiggle_scale=0.01, wick_size=0.01)
    df_1h = _frame(periods=160, start=100.0, drift=0.0, volume_start=60, volume_end=80, wiggle_scale=0.01, wick_size=0.01)

    snapshot = classifier.classify(df_15m, df_1h)

    assert snapshot.regime == RegimeType.DEAD_MARKET
    assert snapshot.allowed_strategies == []
