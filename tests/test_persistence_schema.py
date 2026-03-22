"""Persistence migration tests for structural trade-plan columns."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from core.persistence import SQLitePersistence


def test_initialize_adds_structural_columns_for_existing_tables(tmp_path: Path) -> None:
    """Existing databases should be migrated with structural plan columns."""

    db_path = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id TEXT UNIQUE,
            symbol TEXT NOT NULL,
            strategy TEXT NOT NULL,
            side TEXT NOT NULL,
            status TEXT NOT NULL,
            score REAL,
            expected_value REAL,
            expected_r REAL,
            fees_r REAL,
            slippage_r REAL,
            confidence REAL,
            rationale_json TEXT,
            blockers_json TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            mode TEXT NOT NULL,
            side TEXT NOT NULL,
            status TEXT NOT NULL,
            entry_price REAL,
            mark_price REAL,
            stop_price REAL,
            tp1_price REAL,
            tp2_price REAL,
            quantity REAL,
            leverage REAL,
            used_margin REAL,
            unrealized_pnl REAL,
            realized_pnl REAL,
            strategy TEXT,
            signal_id TEXT,
            metadata_json TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id TEXT UNIQUE,
            symbol TEXT NOT NULL,
            strategy TEXT NOT NULL,
            side TEXT NOT NULL,
            mode TEXT NOT NULL,
            entry_price REAL,
            exit_price REAL,
            stop_price REAL,
            tp1_price REAL,
            tp2_price REAL,
            quantity REAL,
            realized_pnl REAL,
            realized_pnl_usdt REAL,
            pnl_r REAL,
            hold_minutes REAL,
            status TEXT NOT NULL,
            exit_reason TEXT,
            tags_json TEXT,
            rationale_json TEXT,
            created_at TEXT NOT NULL,
            closed_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()

    SQLitePersistence(db_path)

    conn = sqlite3.connect(db_path)
    signal_columns = {row[1] for row in conn.execute("PRAGMA table_info(signals)")}
    position_columns = {row[1] for row in conn.execute("PRAGMA table_info(positions)")}
    trade_columns = {row[1] for row in conn.execute("PRAGMA table_info(trades)")}
    conn.close()

    assert {
        "tp3_price",
        "stop_reason",
        "target_plan_json",
        "rr_to_tp1",
        "trade_rejected_reason",
        "ev_metrics_json",
        "market_regime",
        "chosen_strategy",
        "candidate_strategies_json",
        "rejected_strategies_json",
        "rejection_reasons_json",
        "overlap_score",
        "conflict_resolution_decision",
    } <= signal_columns
    assert {
        "tp3_price",
        "stop_reason",
        "target_plan_json",
        "rr_to_tp2",
        "rr_to_best_target",
        "market_regime",
        "chosen_strategy",
        "candidate_strategies_json",
        "rejected_strategies_json",
        "rejection_reasons_json",
        "overlap_score",
        "conflict_resolution_decision",
    } <= position_columns
    assert {
        "tp3_price",
        "stop_reason",
        "tp1_reason",
        "tp2_reason",
        "tp3_reason",
        "rr_to_tp1",
        "trade_rejected_reason",
        "market_regime",
        "chosen_strategy",
        "candidate_strategies_json",
        "rejected_strategies_json",
        "rejection_reasons_json",
        "overlap_score",
        "conflict_resolution_decision",
    } <= trade_columns
