"""Indicator helpers used by strategy and market-regime modules."""

from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    """Calculate true range."""

    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average true range."""

    return true_range(df).ewm(alpha=1 / period, adjust=False).mean()


def directional_movement(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Positive and negative directional movement."""

    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    return pd.Series(plus_dm, index=df.index), pd.Series(minus_dm, index=df.index)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average directional index."""

    tr = true_range(df)
    plus_dm, minus_dm = directional_movement(df)
    atr_series = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_series.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_series.replace(0, np.nan))
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0.0)


def vwap(df: pd.DataFrame) -> pd.Series:
    """Session VWAP approximation."""

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cumulative_value = (typical_price * df["volume"]).cumsum()
    cumulative_volume = df["volume"].replace(0, np.nan).cumsum()
    return (cumulative_value / cumulative_volume).ffill().fillna(df["close"])


def realized_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Realized volatility using log returns."""

    returns = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    return returns.rolling(window).std(ddof=0) * np.sqrt(window)


def rolling_percentile(series: pd.Series, window: int = 100) -> pd.Series:
    """Rolling percentile rank of the latest value inside each window."""

    def _percentile(values: np.ndarray) -> float:
        current = values[-1]
        rank = (values <= current).sum()
        return rank / len(values)

    return series.rolling(window).apply(_percentile, raw=True).fillna(0.5)


def zscore(series: pd.Series, window: int = 30) -> pd.Series:
    """Rolling z-score."""

    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0).replace(0, np.nan)
    return ((series - mean) / std).fillna(0.0)


def trend_quality(close: pd.Series, window: int = 30) -> pd.Series:
    """Estimate trend quality via rolling linear-regression fit."""

    values = close.to_numpy(dtype=float)
    result = np.full_like(values, fill_value=np.nan)
    x = np.arange(window)
    for idx in range(window - 1, len(values)):
        y = values[idx - window + 1 : idx + 1]
        slope, intercept = np.polyfit(x, y, 1)
        fit = slope * x + intercept
        ss_res = np.sum((y - fit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 0.0 if ss_tot == 0 else max(0.0, 1 - (ss_res / ss_tot))
        result[idx] = np.sign(slope) * r_squared
    return pd.Series(result, index=close.index).fillna(0.0)


def equal_highs_lows(df: pd.DataFrame, lookback: int = 30, tolerance_bps: float = 6.0) -> dict[str, list[float]]:
    """Find approximate equal highs and lows used as liquidity pools."""

    tolerance_multiplier = tolerance_bps / 10_000
    highs = df["high"].tail(lookback)
    lows = df["low"].tail(lookback)
    equal_highs: list[float] = []
    equal_lows: list[float] = []

    high_values = highs.to_list()
    low_values = lows.to_list()
    for idx, value in enumerate(high_values):
        similar = [other for jdx, other in enumerate(high_values) if idx != jdx and abs(other - value) / value <= tolerance_multiplier]
        if similar:
            equal_highs.append(value)
    for idx, value in enumerate(low_values):
        similar = [other for jdx, other in enumerate(low_values) if idx != jdx and abs(other - value) / value <= tolerance_multiplier]
        if similar:
            equal_lows.append(value)

    return {
        "equal_highs": sorted(set(round(value, 8) for value in equal_highs)),
        "equal_lows": sorted(set(round(value, 8) for value in equal_lows)),
    }


def swing_points(df: pd.DataFrame, window: int = 3) -> dict[str, list[tuple[pd.Timestamp, float]]]:
    """Return local swing highs and lows."""

    highs: list[tuple[pd.Timestamp, float]] = []
    lows: list[tuple[pd.Timestamp, float]] = []
    for idx in range(window, len(df) - window):
        center_high = df["high"].iloc[idx]
        center_low = df["low"].iloc[idx]
        if center_high >= df["high"].iloc[idx - window : idx + window + 1].max():
            highs.append((df.index[idx], float(center_high)))
        if center_low <= df["low"].iloc[idx - window : idx + window + 1].min():
            lows.append((df.index[idx], float(center_low)))
    return {"highs": highs, "lows": lows}
