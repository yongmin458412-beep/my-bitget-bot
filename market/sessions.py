"""Trading session helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta

import pandas as pd


@dataclass(slots=True)
class SessionRange:
    """High/low range for a session."""

    name: str
    high: float
    low: float
    start: datetime
    end: datetime


def _session_slice(df: pd.DataFrame, start_hour: int, end_hour: int) -> pd.DataFrame:
    """Slice a UTC-indexed frame by session hour."""

    if df.empty:
        return df
    intraday = df.copy()
    intraday = intraday[(intraday.index.hour >= start_hour) & (intraday.index.hour < end_hour)]
    return intraday


def session_high_low(df: pd.DataFrame, name: str, start_hour: int, end_hour: int) -> SessionRange | None:
    """Return a session range."""

    sliced = _session_slice(df, start_hour, end_hour)
    if sliced.empty:
        return None
    return SessionRange(
        name=name,
        high=float(sliced["high"].max()),
        low=float(sliced["low"].min()),
        start=sliced.index.min().to_pydatetime().replace(tzinfo=UTC),
        end=sliced.index.max().to_pydatetime().replace(tzinfo=UTC),
    )


def previous_day_high_low(df: pd.DataFrame) -> tuple[float | None, float | None]:
    """Return previous UTC-day high and low."""

    if df.empty:
        return None, None
    yesterday = (df.index.max() - timedelta(days=1)).date()
    sliced = df[df.index.date == yesterday]
    if sliced.empty:
        return None, None
    return float(sliced["high"].max()), float(sliced["low"].min())


def asia_session_high_low(df: pd.DataFrame) -> SessionRange | None:
    """Return Asia session high/low."""

    return session_high_low(df, "asia", 0, 8)


def london_session_high_low(df: pd.DataFrame) -> SessionRange | None:
    """Return London session high/low."""

    return session_high_low(df, "london", 7, 16)

