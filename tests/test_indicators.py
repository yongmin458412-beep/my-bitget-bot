"""Indicator tests."""

from __future__ import annotations

import pandas as pd

from market.indicators import adx, atr, vwap


def sample_frame() -> pd.DataFrame:
    """Create a simple OHLCV frame."""

    return pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104, 105, 106, 107],
            "high": [101, 102, 103, 104, 105, 106, 107, 108],
            "low": [99, 100, 101, 102, 103, 104, 105, 106],
            "close": [100.5, 101.5, 102.2, 103.8, 104.4, 105.7, 106.4, 107.1],
            "volume": [10, 12, 13, 15, 18, 21, 24, 26],
        }
    )


def test_atr_returns_series() -> None:
    """ATR should return a same-length series."""

    df = sample_frame()
    result = atr(df, period=3)
    assert len(result) == len(df)
    assert result.iloc[-1] > 0


def test_vwap_is_within_range() -> None:
    """VWAP should be bounded by recent lows/highs."""

    df = sample_frame()
    result = vwap(df)
    assert df["low"].min() <= result.iloc[-1] <= df["high"].max()


def test_adx_non_negative() -> None:
    """ADX should remain non-negative."""

    df = sample_frame()
    result = adx(df, period=3)
    assert result.iloc[-1] >= 0

