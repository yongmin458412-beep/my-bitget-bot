"""Shared dashboard utilities without Streamlit side effects."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sqlite3
import subprocess
from typing import Any
import uuid

import httpx
import pandas as pd
import plotly.graph_objects as go

from core.settings import SettingsManager
from core.utils import deep_merge, dump_json, load_json


ROOT_DIR = Path(__file__).resolve().parents[1]
SETTINGS_MANAGER = SettingsManager()
CONTROL_COMMANDS_PATH = ROOT_DIR / "state" / "control_commands.json"


def load_settings() -> dict[str, Any]:
    """Return public settings snapshot."""

    return SETTINGS_MANAGER.load().to_runtime_dict()


def load_runtime_state() -> dict[str, Any]:
    """Return runtime state JSON."""

    settings = SETTINGS_MANAGER.load()
    return load_json(settings.state_path, default={})


def load_control_commands(path: str | Path | None = None) -> dict[str, Any]:
    """Return queued dashboard control commands."""

    command_path = Path(path) if path is not None else CONTROL_COMMANDS_PATH
    return load_json(command_path, default={"pending": [], "history": []})


def enqueue_control_command(
    action: str,
    *,
    payload: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Append a dashboard control command for the bot runtime to consume."""

    command_path = Path(path) if path is not None else CONTROL_COMMANDS_PATH
    snapshot = load_control_commands(command_path)
    pending = [item for item in snapshot.get("pending", []) if isinstance(item, dict)]
    history = [item for item in snapshot.get("history", []) if isinstance(item, dict)]
    command = {
        "id": uuid.uuid4().hex[:12],
        "action": action,
        "payload": payload or {},
        "created_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "status": "pending",
    }
    pending.append(command)
    dump_json(
        command_path,
        {
            "pending": pending[-50:],
            "history": history[-100:],
        },
    )
    return command


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse ISO timestamps used by runtime state files."""

    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    except ValueError:
        return None


def _bot_processes() -> list[dict[str, Any]]:
    """Return active local bot processes."""

    try:
        output = subprocess.check_output(
            ["ps", "-ax", "-o", "pid=,ppid=,etime=,command="],
            text=True,
        )
    except Exception:
        return []

    processes: list[dict[str, Any]] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or "run_bot.py" not in line:
            continue
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid, ppid, etime, command = parts
        processes.append(
            {
                "pid": pid,
                "ppid": ppid,
                "elapsed": etime,
                "command": command,
            }
        )
    return processes


def inspect_bot_runtime() -> dict[str, Any]:
    """Inspect whether the bot appears to be running and updating state."""

    settings = SETTINGS_MANAGER.load()
    runtime = load_runtime_state()
    processes = _bot_processes()
    now = datetime.now(tz=UTC)
    last_healthcheck_at = _parse_timestamp(runtime.get("last_healthcheck_at"))
    state_mtime = None
    if settings.state_path.exists():
        state_mtime = datetime.fromtimestamp(settings.state_path.stat().st_mtime, tz=UTC)

    health_age_seconds = (now - last_healthcheck_at).total_seconds() if last_healthcheck_at else None
    state_age_seconds = (now - state_mtime).total_seconds() if state_mtime else None
    freshness_limit = max(90, settings.runtime.healthcheck_seconds * 3)
    is_running = bool(processes)
    is_fresh = any(
        age is not None and age <= freshness_limit
        for age in (health_age_seconds, state_age_seconds)
    )
    status = "정상"
    if not is_running:
        status = "중지"
    elif not is_fresh:
        status = "지연"

    return {
        "status": status,
        "is_running": is_running,
        "is_fresh": is_fresh,
        "processes": processes,
        "bot_status": runtime.get("bot_status", "unknown"),
        "paused": runtime.get("paused", False),
        "last_healthcheck_at": runtime.get("last_healthcheck_at"),
        "health_age_seconds": round(health_age_seconds, 1) if health_age_seconds is not None else None,
        "state_file_age_seconds": round(state_age_seconds, 1) if state_age_seconds is not None else None,
        "last_event": runtime.get("last_event", {}),
    }


def run_query(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    """Execute a read-only SQLite query."""

    settings = SETTINGS_MANAGER.load()
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def fetch_public_candles(symbol: str, product_type: str = "USDT-FUTURES", granularity: str = "5m", limit: int = 120) -> pd.DataFrame:
    """Fetch public candles directly for chart rendering."""

    response = httpx.get(
        "https://api.bitget.com/api/v2/mix/market/candles",
        params={
            "symbol": symbol.upper(),
            "productType": product_type,
            "granularity": granularity,
            "limit": limit,
        },
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "quote_volume"])
    numeric_columns = ["open", "high", "low", "close", "volume", "quote_volume"]
    for column in numeric_columns:
        df[column] = df[column].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
    return df.sort_values("timestamp")


def render_candlestick(symbol: str, product_type: str = "USDT-FUTURES") -> go.Figure:
    """Render a simple candlestick chart."""

    df = fetch_public_candles(symbol, product_type=product_type)
    figure = go.Figure()
    if df.empty:
        return figure
    figure.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=symbol,
        )
    )
    figure.update_layout(height=420, margin=dict(l=16, r=16, t=24, b=16))
    return figure


def save_settings_patch(patch: dict[str, Any]) -> None:
    """Merge and save settings overrides."""

    current = load_json(ROOT_DIR / "config" / "bot_settings.json", default={})
    merged = deep_merge(current, patch)
    dump_json(ROOT_DIR / "config" / "bot_settings.json", merged)


def latest_rows(table: str, limit: int = 20) -> list[dict[str, Any]]:
    """Read latest rows from a table."""

    return run_query(f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?", (limit,))
