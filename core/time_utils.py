"""Time and session helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo


UTC_ZONE = ZoneInfo("UTC")
KST_ZONE = ZoneInfo("Asia/Seoul")


@dataclass(slots=True)
class SessionWindow:
    """Named trading session window in UTC."""

    name: str
    start: time
    end: time


ASIA_SESSION = SessionWindow("asia", time(0, 0), time(8, 0))
LONDON_SESSION = SessionWindow("london", time(7, 0), time(16, 0))
NEW_YORK_SESSION = SessionWindow("new_york", time(13, 0), time(22, 0))


def utc_now() -> datetime:
    """Return the current UTC time."""

    return datetime.now(tz=UTC_ZONE)


def kst_now() -> datetime:
    """Return the current Korea time."""

    return datetime.now(tz=KST_ZONE)


def to_kst(dt_value: datetime) -> datetime:
    """Convert a datetime to Korea time."""

    if dt_value.tzinfo is None:
        dt_value = dt_value.replace(tzinfo=UTC)
    return dt_value.astimezone(KST_ZONE)


def to_utc(dt_value: datetime) -> datetime:
    """Convert a datetime to UTC."""

    if dt_value.tzinfo is None:
        dt_value = dt_value.replace(tzinfo=UTC)
    return dt_value.astimezone(UTC_ZONE)


def floor_datetime(dt_value: datetime, minutes: int) -> datetime:
    """Floor a datetime to a minute boundary."""

    if minutes <= 0:
        raise ValueError("minutes must be positive")
    dt_value = to_utc(dt_value)
    discard = timedelta(
        minutes=dt_value.minute % minutes,
        seconds=dt_value.second,
        microseconds=dt_value.microsecond,
    )
    return dt_value - discard


def minutes_between(left: datetime, right: datetime) -> float:
    """Return total minutes between two datetimes."""

    return abs((to_utc(right) - to_utc(left)).total_seconds()) / 60.0


def is_within_window(reference: datetime, center: datetime, minutes: int) -> bool:
    """Check whether a time is within a symmetric minute window."""

    return minutes_between(reference, center) <= minutes


def within_session(dt_value: datetime, session: SessionWindow) -> bool:
    """Check whether a UTC datetime is inside a named session."""

    dt_value = to_utc(dt_value)
    current = dt_value.time()
    return session.start <= current < session.end


def minutes_until_next_boundary(dt_value: datetime, interval_minutes: int) -> int:
    """Return minutes until the next boundary for a timeframe."""

    floored = floor_datetime(dt_value, interval_minutes)
    boundary = floored + timedelta(minutes=interval_minutes)
    remaining = boundary - to_utc(dt_value)
    return max(0, int(remaining.total_seconds() // 60))

