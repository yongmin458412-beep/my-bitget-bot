"""Tests for FillHandler verification and persistence logic."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from core.persistence import SQLitePersistence
from core.state_store import StateStore
from execution.fill_handler import FillHandler
from journal.trade_journal import TradeJournal


def _make_handler(tmp_path: Path) -> tuple[FillHandler, StateStore, SQLitePersistence]:
    persistence = SQLitePersistence(tmp_path / "test.sqlite3")
    state_store = StateStore(tmp_path / "state.json", persistence)
    journal = TradeJournal(persistence)
    handler = FillHandler(persistence, state_store, journal)
    return handler, state_store, persistence


def _valid_fill() -> dict:
    now = datetime.now(tz=UTC).isoformat(timespec="seconds")
    return {
        "client_order_id": "entry-test-001",
        "exchange_order_id": "ex-001",
        "symbol": "BTCUSDT",
        "mode": "DEMO",
        "side": "long",
        "order_type": "limit",
        "price": 100.0,
        "quantity": 0.01,
        "filled_quantity": 0.01,
        "avg_fill_price": 100.1,
        "stop_price": 98.0,
        "tp1_price": 102.0,
        "tp2_price": 104.0,
        "tp3_price": 106.0,
        "strategy": "break_retest",
        "signal_id": "sig-abc123",
        "leverage": 10.0,
        "reason": "unit_test",
        "created_at": now,
        "updated_at": now,
    }


def test_fill_handler_accepts_valid_fill(tmp_path: Path) -> None:
    """Valid fill should persist order + position + state update."""

    handler, state_store, persistence = _make_handler(tmp_path)
    payload = _valid_fill()
    handler.on_entry_filled(payload)

    # Position should appear in state
    assert "BTCUSDT" in state_store.state.open_positions

    # Order should be in DB
    row = persistence.fetchone(
        "SELECT status FROM orders WHERE client_order_id = ?",
        ("entry-test-001",),
    )
    assert row is not None
    assert row["status"] == "filled"

    # Position should be in DB
    pos = persistence.fetchone(
        "SELECT status, entry_price FROM positions WHERE symbol = ?",
        ("BTCUSDT",),
    )
    assert pos is not None
    assert pos["status"] == "open"
    assert pos["entry_price"] == 100.1  # avg_fill_price used


def test_fill_handler_rejects_zero_quantity(tmp_path: Path) -> None:
    """Fill with zero quantity should not persist anything."""

    handler, state_store, _ = _make_handler(tmp_path)
    payload = _valid_fill()
    payload["filled_quantity"] = 0.0
    payload["quantity"] = 0.0

    handler.on_entry_filled(payload)

    assert "BTCUSDT" not in state_store.state.open_positions


def test_fill_handler_rejects_zero_price(tmp_path: Path) -> None:
    """Fill with zero avg_fill_price should not persist anything."""

    handler, state_store, _ = _make_handler(tmp_path)
    payload = _valid_fill()
    payload["avg_fill_price"] = 0.0
    payload["price"] = 0.0

    handler.on_entry_filled(payload)

    assert "BTCUSDT" not in state_store.state.open_positions


def test_fill_handler_warns_on_quantity_mismatch(tmp_path: Path, capsys) -> None:
    """Fill where filled != requested (>1%) should still persist but log a warning."""

    handler, state_store, _ = _make_handler(tmp_path)
    payload = _valid_fill()
    payload["quantity"] = 0.01
    payload["filled_quantity"] = 0.005  # 50% mismatch

    handler.on_entry_filled(payload)

    # Should still persist (mismatch is a warning, not rejection)
    assert "BTCUSDT" in state_store.state.open_positions


def test_fill_handler_on_position_closed_removes_state(tmp_path: Path) -> None:
    """Closing fill should remove position from runtime state."""

    handler, state_store, _ = _make_handler(tmp_path)
    payload = _valid_fill()
    handler.on_entry_filled(payload)

    assert "BTCUSDT" in state_store.state.open_positions

    close_payload = {
        "symbol": "BTCUSDT",
        "mode": "DEMO",
        "side": "long",
        "exit_price": 103.0,
        "realized_pnl": 0.03,
        "exit_reason": "partial_tp1",
        "client_order_id": "exit-001",
    }
    handler.on_position_closed(close_payload)

    assert "BTCUSDT" not in state_store.state.open_positions
