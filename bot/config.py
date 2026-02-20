from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import time

KST = timezone(timedelta(hours=9))


@dataclass(frozen=True)
class BotPaths:
    settings_file: str = "bot_settings.json"
    runtime_file: str = "runtime_state.json"
    log_file: str = "trade_log.csv"
    monitor_file: str = "monitor_state.json"
    sqlite_db_file: str = "bot_data.db"
    detail_dir: str = "trade_details"
    daily_report_dir: str = "daily_reports"
    event_image_dir: str = "trade_event_images"


def now_kst() -> datetime:
    return datetime.now(KST)


def now_kst_str() -> str:
    return now_kst().strftime("%Y-%m-%d %H:%M:%S")


def today_kst_str() -> str:
    return now_kst().strftime("%Y-%m-%d")


def next_midnight_kst_epoch() -> float:
    try:
        dt = now_kst()
        dt2 = (dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return float(dt2.timestamp())
    except Exception:
        return float(time.time() + 60 * 60 * 6)


def parse_time_kst(s: str) -> datetime | None:
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)
    except Exception:
        return None


def dt_to_epoch(dt: datetime) -> float:
    try:
        return float(dt.timestamp())
    except Exception:
        return float(time.time())


def epoch_to_kst_str(epoch: float) -> str:
    try:
        return datetime.fromtimestamp(float(epoch), tz=KST).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return now_kst_str()
