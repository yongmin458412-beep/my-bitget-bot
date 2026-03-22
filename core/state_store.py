"""Runtime state persistence and recovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .persistence import SQLitePersistence
from .logger import get_logger
from .utils import dump_json, ensure_directory, load_json


@dataclass(slots=True)
class RuntimeState:
    """Serializable runtime state used by the bot and UI."""

    bot_status: str = "starting"
    paused: bool = False
    last_reload_at: str | None = None
    last_healthcheck_at: str | None = None
    active_universe: list[str] = field(default_factory=list)
    tracked_symbols: list[str] = field(default_factory=list)
    open_positions: dict[str, dict[str, Any]] = field(default_factory=dict)
    open_orders: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending_reasons: dict[str, str] = field(default_factory=dict)
    risk_flags: list[str] = field(default_factory=list)
    last_event: dict[str, Any] = field(default_factory=dict)


class StateStore:
    """Manage runtime state in memory, JSON, and SQLite."""

    def __init__(self, state_path: str | Path, persistence: SQLitePersistence) -> None:
        self.state_path = Path(state_path)
        ensure_directory(self.state_path.parent)
        self.persistence = persistence
        self.state = RuntimeState()
        self.logger = get_logger(__name__)

    def load(self) -> RuntimeState:
        """Restore state from disk or SQLite."""

        raw = load_json(self.state_path, default={})
        if raw:
            self.state = RuntimeState(**raw)
        else:
            open_positions = {
                row["symbol"]: row
                for row in self.persistence.fetchall(
                    "SELECT * FROM positions WHERE status = 'open'"
                )
            }
            open_orders = {
                row["client_order_id"]: row
                for row in self.persistence.fetchall(
                    "SELECT * FROM orders WHERE status IN ('new', 'open', 'partially_filled')"
                )
            }
            self.state.open_positions = open_positions
            self.state.open_orders = open_orders
        return self.state

    def save(self) -> None:
        """Persist current state to disk (atomic tmp+rename) and SQLite."""

        payload = asdict(self.state)
        tmp_path = self.state_path.with_suffix(".tmp")
        try:
            dump_json(tmp_path, payload)
            tmp_path.replace(self.state_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        try:
            self.persistence.upsert_runtime_value(
                "runtime_state",
                payload,
                datetime.now(tz=UTC).isoformat(timespec="seconds"),
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("SQLite state save failed (JSON saved OK)", extra={"extra_data": {"error": str(exc)}})

    def set_status(self, status: str) -> None:
        """Update top-level bot status."""

        self.state.bot_status = status
        self.save()

    def set_paused(self, paused: bool) -> None:
        """Update pause state."""

        self.state.paused = paused
        self.save()

    def set_active_universe(self, symbols: list[str]) -> None:
        """Update active universe."""

        self.state.active_universe = symbols
        self.save()

    def set_tracked_symbols(self, symbols: list[str]) -> None:
        """Update tracked symbol list."""

        self.state.tracked_symbols = symbols
        self.save()

    def update_position(self, symbol: str, payload: dict[str, Any]) -> None:
        """Insert or replace a runtime position."""

        self.state.open_positions[symbol] = payload
        self.save()

    def remove_position(self, symbol: str) -> None:
        """Remove a runtime position."""

        self.state.open_positions.pop(symbol, None)
        self.save()

    def update_order(self, client_order_id: str, payload: dict[str, Any]) -> None:
        """Insert or replace a runtime order."""

        self.state.open_orders[client_order_id] = payload
        self.save()

    def remove_order(self, client_order_id: str) -> None:
        """Remove a runtime order."""

        self.state.open_orders.pop(client_order_id, None)
        self.save()

    def set_risk_flags(self, flags: list[str]) -> None:
        """Update active risk flags."""

        self.state.risk_flags = flags
        self.save()

    def set_last_event(self, title: str, message: str, level: str = "INFO") -> None:
        """Update the last noteworthy event."""

        self.state.last_event = {
            "title": title,
            "message": message,
            "level": level,
            "timestamp": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        }
        self.save()

