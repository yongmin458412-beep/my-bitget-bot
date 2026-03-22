"""Reusable confirmation primitives."""

from __future__ import annotations

import pandas as pd


def bullish_confirmation(df: pd.DataFrame) -> bool:
    """Return True when the latest candle shows bullish confirmation."""

    if len(df) < 2:
        return False
    candle = df.iloc[-1]
    body = candle["close"] - candle["open"]
    upper_wick = candle["high"] - candle["close"]
    return body > 0 and upper_wick <= abs(body) * 0.6


def bearish_confirmation(df: pd.DataFrame) -> bool:
    """Return True when the latest candle shows bearish confirmation."""

    if len(df) < 2:
        return False
    candle = df.iloc[-1]
    body = candle["open"] - candle["close"]
    lower_wick = candle["close"] - candle["low"]
    return body > 0 and lower_wick <= abs(body) * 0.6


def volume_recovered(df: pd.DataFrame, multiple: float = 1.05) -> bool:
    """Return True when the latest volume exceeds rolling average."""

    if len(df) < 20:
        return False
    baseline = df["volume"].iloc[-20:-1].mean()
    return float(df["volume"].iloc[-1]) >= float(baseline * multiple)


def mss_bullish(df: pd.DataFrame) -> bool:
    """Mini market-structure shift for long setups."""

    if len(df) < 6:
        return False
    recent_high = float(df["high"].iloc[-6:-2].max())
    return float(df["close"].iloc[-1]) > recent_high


def mss_bearish(df: pd.DataFrame) -> bool:
    """Mini market-structure shift for short setups."""

    if len(df) < 6:
        return False
    recent_low = float(df["low"].iloc[-6:-2].min())
    return float(df["close"].iloc[-1]) < recent_low


def find_fvg_zone(df: pd.DataFrame) -> tuple[float, float] | None:
    """Find a simple fair-value gap zone near the latest candles."""

    if len(df) < 3:
        return None
    left = df.iloc[-3]
    right = df.iloc[-1]
    if float(left["high"]) < float(right["low"]):
        return float(left["high"]), float(right["low"])
    if float(left["low"]) > float(right["high"]):
        return float(right["high"]), float(left["low"])
    return None

