"""Tests for StateStore atomic write and recovery logic."""

from __future__ import annotations

import json
from pathlib import Path

from core.persistence import SQLitePersistence
from core.state_store import StateStore


def _make_store(tmp_path: Path) -> tuple[StateStore, Path]:
    state_path = tmp_path / "state.json"
    persistence = SQLitePersistence(tmp_path / "db.sqlite3")
    store = StateStore(state_path, persistence)
    return store, state_path


def test_save_creates_json_file(tmp_path: Path) -> None:
    """save() should create a readable JSON file."""

    store, state_path = _make_store(tmp_path)
    store.set_status("running")

    assert state_path.exists()
    data = json.loads(state_path.read_text())
    assert data["bot_status"] == "running"


def test_save_no_tmp_file_left_behind(tmp_path: Path) -> None:
    """After successful save, no .tmp file should remain."""

    store, state_path = _make_store(tmp_path)
    store.set_status("running")

    tmp_file = state_path.with_suffix(".tmp")
    assert not tmp_file.exists()


def test_load_from_json_restores_state(tmp_path: Path) -> None:
    """load() should restore state previously saved to JSON."""

    store, state_path = _make_store(tmp_path)
    store.update_position("BTCUSDT", {"symbol": "BTCUSDT", "side": "long", "entry_price": 100.0})
    store.set_status("running")

    # Re-create store from same path
    persistence2 = SQLitePersistence(tmp_path / "db.sqlite3")
    store2 = StateStore(state_path, persistence2)
    store2.load()

    assert store2.state.bot_status == "running"
    assert "BTCUSDT" in store2.state.open_positions


def test_load_fallback_to_sqlite_when_json_empty(tmp_path: Path) -> None:
    """If JSON is missing/empty, load() should fall back to SQLite."""

    state_path = tmp_path / "state.json"
    persistence = SQLitePersistence(tmp_path / "db.sqlite3")

    # Put an open order in DB directly
    persistence.save_order({
        "client_order_id": "oid-1",
        "exchange_order_id": "ex-1",
        "symbol": "ETHUSDT",
        "mode": "DEMO",
        "side": "long",
        "order_type": "limit",
        "status": "open",
        "price": 2000.0,
        "quantity": 0.1,
        "filled_quantity": 0.0,
        "avg_fill_price": None,
        "reduce_only": False,
        "reason": "test",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    })

    # No JSON file → should load from SQLite
    store = StateStore(state_path, persistence)
    store.load()

    assert "oid-1" in store.state.open_orders


def test_update_and_remove_position(tmp_path: Path) -> None:
    """update_position / remove_position should mutate state and persist."""

    store, state_path = _make_store(tmp_path)
    store.update_position("SOLUSDT", {"symbol": "SOLUSDT", "side": "short"})
    assert "SOLUSDT" in store.state.open_positions

    store.remove_position("SOLUSDT")
    assert "SOLUSDT" not in store.state.open_positions

    # Verify JSON reflects removal
    data = json.loads(state_path.read_text())
    assert "SOLUSDT" not in data["open_positions"]


def test_set_risk_flags(tmp_path: Path) -> None:
    """Risk flags should be saved and loadable."""

    store, state_path = _make_store(tmp_path)
    store.set_risk_flags(["daily_loss_limit", "news_block"])

    data = json.loads(state_path.read_text())
    assert "daily_loss_limit" in data["risk_flags"]
    assert "news_block" in data["risk_flags"]
