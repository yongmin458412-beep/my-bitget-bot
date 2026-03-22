"""Trade journal tests for structural fields."""

from __future__ import annotations

from pathlib import Path

from core.persistence import SQLitePersistence
from journal.trade_journal import TradeJournal


def test_trade_journal_preserves_structural_fields(tmp_path: Path) -> None:
    """Entry/exit journaling should retain structural stop/target details."""

    persistence = SQLitePersistence(tmp_path / "journal.sqlite3")
    journal = TradeJournal(persistence)
    entry_payload = {
        "trade_id": "sig-1",
        "signal_id": "sig-1",
        "symbol": "BTCUSDT",
        "strategy": "break_retest",
        "side": "long",
        "mode": "DEMO",
        "price": 100.0,
        "avg_fill_price": 100.0,
        "stop_price": 99.0,
        "tp1_price": 102.0,
        "tp2_price": 104.0,
        "tp3_price": 106.0,
        "filled_quantity": 1.0,
        "stop_reason": "retest_low_atr_buffer",
        "target_plan": [
            {"price": 102.0, "reason": "previous_high", "priority": 1},
            {"price": 104.0, "reason": "session_high", "priority": 2},
            {"price": 106.0, "reason": "range_opposite_side", "priority": 3},
        ],
        "rr_to_tp1": 2.0,
        "rr_to_tp2": 4.0,
        "rr_to_best_target": 6.0,
        "ev_metrics": {"expected_value": 1.2},
        "created_at": "2026-03-18T00:00:00+00:00",
    }
    journal.log_entry(entry_payload)
    journal.log_exit(
        {
            "symbol": "BTCUSDT",
            "mode": "DEMO",
            "side": "long",
            "price": 106.0,
            "avg_fill_price": 106.0,
            "realized_pnl": 6.0,
            "realized_pnl_usdt": 6.0,
            "pnl_r": 6.0,
            "exit_reason": "tp3_runner",
        }
    )

    latest = journal.why_symbol("BTCUSDT")
    assert latest["trade"]["stop_reason"] == "retest_low_atr_buffer"
    assert "previous_high" in latest["rule_summary"]["target_reasons"]
    assert "TP1 RR" in latest["rule_summary"]["rr_commentary"]
