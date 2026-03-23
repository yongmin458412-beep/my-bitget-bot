"""SQLite persistence layer for signals, orders, trades, and runtime events."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from .utils import ensure_directory


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS trades (
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
        tp3_price REAL,
        quantity REAL,
        realized_pnl REAL,
        realized_pnl_usdt REAL,
        pnl_r REAL,
        rr_to_tp1 REAL,
        rr_to_tp2 REAL,
        rr_to_best_target REAL,
        market_regime TEXT,
        chosen_strategy TEXT,
        candidate_strategies_json TEXT,
        rejected_strategies_json TEXT,
        rejection_reasons_json TEXT,
        overlap_score REAL,
        conflict_resolution_decision TEXT,
        hold_minutes REAL,
        status TEXT NOT NULL,
        exit_reason TEXT,
        stop_reason TEXT,
        tp1_reason TEXT,
        tp2_reason TEXT,
        tp3_reason TEXT,
        trade_rejected_reason TEXT,
        ev_metrics_json TEXT,
        tags_json TEXT,
        rationale_json TEXT,
        created_at TEXT NOT NULL,
        closed_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        client_order_id TEXT UNIQUE,
        exchange_order_id TEXT,
        symbol TEXT NOT NULL,
        mode TEXT NOT NULL,
        side TEXT NOT NULL,
        order_type TEXT NOT NULL,
        status TEXT NOT NULL,
        price REAL,
        quantity REAL,
        filled_quantity REAL DEFAULT 0,
        avg_fill_price REAL,
        reduce_only INTEGER DEFAULT 0,
        reason TEXT,
        payload_json TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS positions (
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
        tp3_price REAL,
        stop_reason TEXT,
        target_plan_json TEXT,
        rr_to_tp1 REAL,
        rr_to_tp2 REAL,
        rr_to_best_target REAL,
        market_regime TEXT,
        chosen_strategy TEXT,
        candidate_strategies_json TEXT,
        rejected_strategies_json TEXT,
        rejection_reasons_json TEXT,
        overlap_score REAL,
        conflict_resolution_decision TEXT,
        quantity REAL,
        leverage REAL,
        used_margin REAL,
        unrealized_pnl REAL,
        realized_pnl REAL,
        strategy TEXT,
        signal_id TEXT,
        metadata_json TEXT,
        updated_at TEXT NOT NULL,
        UNIQUE(symbol, mode, side, status)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_id TEXT UNIQUE,
        symbol TEXT NOT NULL,
        strategy TEXT NOT NULL,
        side TEXT NOT NULL,
        status TEXT NOT NULL,
        entry_price REAL,
        stop_price REAL,
        tp1_price REAL,
        tp2_price REAL,
        tp3_price REAL,
        score REAL,
        expected_value REAL,
        expected_r REAL,
        rr_to_tp1 REAL,
        rr_to_tp2 REAL,
        rr_to_best_target REAL,
        market_regime TEXT,
        chosen_strategy TEXT,
        candidate_strategies_json TEXT,
        rejected_strategies_json TEXT,
        rejection_reasons_json TEXT,
        overlap_score REAL,
        conflict_resolution_decision TEXT,
        fees_r REAL,
        slippage_r REAL,
        confidence REAL,
        stop_reason TEXT,
        target_plan_json TEXT,
        trade_rejected_reason TEXT,
        ev_metrics_json TEXT,
        rationale_json TEXT,
        blockers_json TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS symbol_scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        product_type TEXT NOT NULL,
        total_score REAL NOT NULL,
        liquidity_score REAL,
        volatility_score REAL,
        spread_score REAL,
        depth_score REAL,
        trend_score REAL,
        anomaly_score REAL,
        age_score REAL,
        tradability_score REAL,
        metrics_json TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS news_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        news_hash TEXT UNIQUE,
        source TEXT NOT NULL,
        title TEXT NOT NULL,
        url TEXT,
        published_at TEXT,
        content TEXT,
        related_assets_json TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ai_news_analysis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        news_hash TEXT NOT NULL,
        summary_ko TEXT NOT NULL,
        impacted_assets_json TEXT,
        impact_level TEXT,
        direction_bias TEXT,
        validity_window_minutes INTEGER,
        confidence REAL,
        event_type TEXT,
        should_block_new_entries INTEGER,
        notes TEXT,
        created_at TEXT NOT NULL,
        UNIQUE(news_hash)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS daily_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_date TEXT UNIQUE,
        mode TEXT NOT NULL,
        realized_pnl REAL DEFAULT 0,
        unrealized_pnl REAL DEFAULT 0,
        pnl_r REAL DEFAULT 0,
        fees REAL DEFAULT 0,
        trade_count INTEGER DEFAULT 0,
        win_count INTEGER DEFAULT 0,
        loss_count INTEGER DEFAULT 0,
        max_drawdown REAL DEFAULT 0,
        metadata_json TEXT,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS bot_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT NOT NULL,
        level TEXT NOT NULL,
        title TEXT NOT NULL,
        message TEXT NOT NULL,
        metadata_json TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS settings_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mode TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runtime_kv (
        key TEXT PRIMARY KEY,
        value_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
]

INDEX_STATEMENTS = [
    "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy)",
    "CREATE INDEX IF NOT EXISTS idx_trades_created_at ON trades(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)",
    "CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)",
    "CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_positions_symbol_status ON positions(symbol, status)",
    "CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals(strategy)",
    "CREATE INDEX IF NOT EXISTS idx_signals_created_at ON signals(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_bot_events_created_at ON bot_events(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_news_items_created_at ON news_items(created_at)",
]

TABLE_MIGRATIONS: dict[str, dict[str, str]] = {
    "signals": {
        "entry_price": "REAL",
        "stop_price": "REAL",
        "tp1_price": "REAL",
        "tp2_price": "REAL",
        "tp3_price": "REAL",
        "rr_to_tp1": "REAL",
        "rr_to_tp2": "REAL",
        "rr_to_best_target": "REAL",
        "market_regime": "TEXT",
        "chosen_strategy": "TEXT",
        "candidate_strategies_json": "TEXT",
        "rejected_strategies_json": "TEXT",
        "rejection_reasons_json": "TEXT",
        "overlap_score": "REAL",
        "conflict_resolution_decision": "TEXT",
        "stop_reason": "TEXT",
        "target_plan_json": "TEXT",
        "trade_rejected_reason": "TEXT",
        "ev_metrics_json": "TEXT",
    },
    "trades": {
        "tp3_price": "REAL",
        "rr_to_tp1": "REAL",
        "rr_to_tp2": "REAL",
        "rr_to_best_target": "REAL",
        "market_regime": "TEXT",
        "chosen_strategy": "TEXT",
        "candidate_strategies_json": "TEXT",
        "rejected_strategies_json": "TEXT",
        "rejection_reasons_json": "TEXT",
        "overlap_score": "REAL",
        "conflict_resolution_decision": "TEXT",
        "stop_reason": "TEXT",
        "tp1_reason": "TEXT",
        "tp2_reason": "TEXT",
        "tp3_reason": "TEXT",
        "trade_rejected_reason": "TEXT",
        "ev_metrics_json": "TEXT",
    },
    "positions": {
        "tp3_price": "REAL",
        "stop_reason": "TEXT",
        "target_plan_json": "TEXT",
        "rr_to_tp1": "REAL",
        "rr_to_tp2": "REAL",
        "rr_to_best_target": "REAL",
        "market_regime": "TEXT",
        "chosen_strategy": "TEXT",
        "candidate_strategies_json": "TEXT",
        "rejected_strategies_json": "TEXT",
        "rejection_reasons_json": "TEXT",
        "overlap_score": "REAL",
        "conflict_resolution_decision": "TEXT",
    },
}


class SQLitePersistence:
    """Simple SQLite wrapper with thread-safe access."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        ensure_directory(self.db_path.parent)
        self._lock = threading.RLock()
        self.initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a SQLite connection with row access."""

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def initialize(self) -> None:
        """Create tables if they do not exist."""

        with self.connection() as conn:
            for statement in SCHEMA_STATEMENTS:
                conn.execute(statement)
            self._migrate_columns(conn)
            for index_stmt in INDEX_STATEMENTS:
                conn.execute(index_stmt)

    def _migrate_columns(self, conn: sqlite3.Connection) -> None:
        """Add newly introduced columns to existing tables."""

        for table_name, columns in TABLE_MIGRATIONS.items():
            existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table_name})")}
            for column_name, column_type in columns.items():
                if column_name in existing:
                    continue
                conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        """Execute a write query."""

        with self.connection() as conn:
            conn.execute(query, params)

    def executemany(self, query: str, params: Iterable[tuple[Any, ...]]) -> None:
        """Execute a batch write query."""

        with self.connection() as conn:
            conn.executemany(query, params)

    def fetchall(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Fetch multiple rows as dictionaries."""

        with self.connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def fetchone(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        """Fetch a single row as a dictionary."""

        with self.connection() as conn:
            cursor = conn.execute(query, params)
            row = cursor.fetchone()
            return dict(row) if row else None

    def upsert_runtime_value(self, key: str, value: Any, updated_at: str) -> None:
        """Store a small runtime snapshot."""

        self.execute(
            """
            INSERT INTO runtime_kv (key, value_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at
            """,
            (key, json.dumps(value, ensure_ascii=False), updated_at),
        )

    def get_runtime_value(self, key: str) -> Any | None:
        """Read a runtime snapshot."""

        row = self.fetchone("SELECT value_json FROM runtime_kv WHERE key = ?", (key,))
        if not row:
            return None
        return json.loads(row["value_json"])

    def insert_event(
        self,
        event_type: str,
        level: str,
        title: str,
        message: str,
        metadata: dict[str, Any] | None,
        created_at: str,
    ) -> None:
        """Persist a bot event."""

        self.execute(
            """
            INSERT INTO bot_events (event_type, level, title, message, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event_type,
                level,
                title,
                message,
                json.dumps(metadata or {}, ensure_ascii=False),
                created_at,
            ),
        )

    def save_signal(self, payload: dict[str, Any]) -> None:
        """Persist a signal snapshot."""

        self.execute(
            """
            INSERT OR REPLACE INTO signals (
                signal_id, symbol, strategy, side, status, entry_price, stop_price,
                tp1_price, tp2_price, tp3_price, score, expected_value, expected_r,
                rr_to_tp1, rr_to_tp2, rr_to_best_target, market_regime, chosen_strategy,
                candidate_strategies_json, rejected_strategies_json, rejection_reasons_json,
                overlap_score, conflict_resolution_decision, fees_r, slippage_r,
                confidence, stop_reason, target_plan_json, trade_rejected_reason,
                ev_metrics_json, rationale_json, blockers_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["signal_id"],
                payload["symbol"],
                payload["strategy"],
                payload["side"],
                payload["status"],
                payload.get("entry_price"),
                payload.get("stop_price"),
                payload.get("tp1_price"),
                payload.get("tp2_price"),
                payload.get("tp3_price"),
                payload.get("score"),
                payload.get("expected_value"),
                payload.get("expected_r"),
                payload.get("rr_to_tp1"),
                payload.get("rr_to_tp2"),
                payload.get("rr_to_best_target"),
                payload.get("market_regime"),
                payload.get("chosen_strategy"),
                json.dumps(payload.get("candidate_strategies", []), ensure_ascii=False),
                json.dumps(payload.get("rejected_strategies", []), ensure_ascii=False),
                json.dumps(payload.get("rejection_reasons", []), ensure_ascii=False),
                payload.get("overlap_score", 0.0),
                payload.get("conflict_resolution_decision", ""),
                payload.get("fees_r"),
                payload.get("slippage_r"),
                payload.get("confidence"),
                payload.get("stop_reason"),
                json.dumps(payload.get("target_plan", []), ensure_ascii=False),
                payload.get("trade_rejected_reason"),
                json.dumps(payload.get("ev_metrics", {}), ensure_ascii=False),
                json.dumps(payload.get("rationale", {}), ensure_ascii=False),
                json.dumps(payload.get("blockers", []), ensure_ascii=False),
                payload["created_at"],
            ),
        )

    def save_order(self, payload: dict[str, Any]) -> None:
        """Persist order state."""

        self.execute(
            """
            INSERT INTO orders (
                client_order_id, exchange_order_id, symbol, mode, side, order_type, status,
                price, quantity, filled_quantity, avg_fill_price, reduce_only, reason,
                payload_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(client_order_id) DO UPDATE SET
                exchange_order_id=excluded.exchange_order_id,
                status=excluded.status,
                filled_quantity=excluded.filled_quantity,
                avg_fill_price=excluded.avg_fill_price,
                reason=excluded.reason,
                payload_json=excluded.payload_json,
                updated_at=excluded.updated_at
            """,
            (
                payload["client_order_id"],
                payload.get("exchange_order_id"),
                payload["symbol"],
                payload["mode"],
                payload["side"],
                payload["order_type"],
                payload["status"],
                payload.get("price"),
                payload.get("quantity"),
                payload.get("filled_quantity", 0.0),
                payload.get("avg_fill_price"),
                int(payload.get("reduce_only", False)),
                payload.get("reason"),
                json.dumps(payload, ensure_ascii=False),
                payload["created_at"],
                payload["updated_at"],
            ),
        )

    def save_position(self, payload: dict[str, Any]) -> None:
        """Persist position state."""

        self.execute(
            """
            INSERT INTO positions (
                symbol, mode, side, status, entry_price, mark_price, stop_price, tp1_price,
                tp2_price, tp3_price, stop_reason, target_plan_json, rr_to_tp1, rr_to_tp2,
                rr_to_best_target, market_regime, chosen_strategy, candidate_strategies_json,
                rejected_strategies_json, rejection_reasons_json, overlap_score,
                conflict_resolution_decision, quantity, leverage, used_margin, unrealized_pnl, realized_pnl,
                strategy, signal_id, metadata_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, mode, side, status) DO UPDATE SET
                entry_price=excluded.entry_price,
                mark_price=excluded.mark_price,
                stop_price=excluded.stop_price,
                tp1_price=excluded.tp1_price,
                tp2_price=excluded.tp2_price,
                tp3_price=excluded.tp3_price,
                stop_reason=excluded.stop_reason,
                target_plan_json=excluded.target_plan_json,
                rr_to_tp1=excluded.rr_to_tp1,
                rr_to_tp2=excluded.rr_to_tp2,
                rr_to_best_target=excluded.rr_to_best_target,
                market_regime=excluded.market_regime,
                chosen_strategy=excluded.chosen_strategy,
                candidate_strategies_json=excluded.candidate_strategies_json,
                rejected_strategies_json=excluded.rejected_strategies_json,
                rejection_reasons_json=excluded.rejection_reasons_json,
                overlap_score=excluded.overlap_score,
                conflict_resolution_decision=excluded.conflict_resolution_decision,
                quantity=excluded.quantity,
                leverage=excluded.leverage,
                used_margin=excluded.used_margin,
                unrealized_pnl=excluded.unrealized_pnl,
                realized_pnl=excluded.realized_pnl,
                strategy=excluded.strategy,
                signal_id=excluded.signal_id,
                metadata_json=excluded.metadata_json,
                updated_at=excluded.updated_at
            """,
            (
                payload["symbol"],
                payload["mode"],
                payload["side"],
                payload["status"],
                payload.get("entry_price"),
                payload.get("mark_price"),
                payload.get("stop_price"),
                payload.get("tp1_price"),
                payload.get("tp2_price"),
                payload.get("tp3_price"),
                payload.get("stop_reason"),
                json.dumps(payload.get("target_plan", []), ensure_ascii=False),
                payload.get("rr_to_tp1"),
                payload.get("rr_to_tp2"),
                payload.get("rr_to_best_target"),
                payload.get("market_regime"),
                payload.get("chosen_strategy"),
                json.dumps(payload.get("candidate_strategies", []), ensure_ascii=False),
                json.dumps(payload.get("rejected_strategies", []), ensure_ascii=False),
                json.dumps(payload.get("rejection_reasons", []), ensure_ascii=False),
                payload.get("overlap_score", 0.0),
                payload.get("conflict_resolution_decision", ""),
                payload.get("quantity"),
                payload.get("leverage"),
                payload.get("used_margin"),
                payload.get("unrealized_pnl"),
                payload.get("realized_pnl"),
                payload.get("strategy"),
                payload.get("signal_id"),
                json.dumps(payload.get("metadata", {}), ensure_ascii=False),
                payload["updated_at"],
            ),
        )

    def close_position(self, symbol: str, mode: str, side: str, updated_at: str) -> None:
        """Mark an open position as closed."""

        self.execute(
            """
            UPDATE positions
            SET status = 'closed', updated_at = ?
            WHERE symbol = ? AND mode = ? AND side = ? AND status = 'open'
            """,
            (updated_at, symbol, mode, side),
        )

    def save_trade(self, payload: dict[str, Any]) -> None:
        """Persist a trade journal row."""

        self.execute(
            """
            INSERT OR REPLACE INTO trades (
                trade_id, symbol, strategy, side, mode, entry_price, exit_price, stop_price,
                tp1_price, tp2_price, tp3_price, quantity, realized_pnl, realized_pnl_usdt, pnl_r,
                rr_to_tp1, rr_to_tp2, rr_to_best_target, market_regime, chosen_strategy,
                candidate_strategies_json, rejected_strategies_json, rejection_reasons_json,
                overlap_score, conflict_resolution_decision, hold_minutes, status, exit_reason,
                stop_reason, tp1_reason, tp2_reason, tp3_reason, trade_rejected_reason,
                ev_metrics_json, tags_json, rationale_json, created_at, closed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["trade_id"],
                payload["symbol"],
                payload["strategy"],
                payload["side"],
                payload["mode"],
                payload.get("entry_price"),
                payload.get("exit_price"),
                payload.get("stop_price"),
                payload.get("tp1_price"),
                payload.get("tp2_price"),
                payload.get("tp3_price"),
                payload.get("quantity"),
                payload.get("realized_pnl"),
                payload.get("realized_pnl_usdt"),
                payload.get("pnl_r"),
                payload.get("rr_to_tp1"),
                payload.get("rr_to_tp2"),
                payload.get("rr_to_best_target"),
                payload.get("market_regime"),
                payload.get("chosen_strategy"),
                json.dumps(payload.get("candidate_strategies", []), ensure_ascii=False),
                json.dumps(payload.get("rejected_strategies", []), ensure_ascii=False),
                json.dumps(payload.get("rejection_reasons", []), ensure_ascii=False),
                payload.get("overlap_score", 0.0),
                payload.get("conflict_resolution_decision", ""),
                payload.get("hold_minutes"),
                payload["status"],
                payload.get("exit_reason"),
                payload.get("stop_reason"),
                payload.get("tp1_reason"),
                payload.get("tp2_reason"),
                payload.get("tp3_reason"),
                payload.get("trade_rejected_reason"),
                json.dumps(payload.get("ev_metrics", {}), ensure_ascii=False),
                json.dumps(payload.get("tags", []), ensure_ascii=False),
                json.dumps(payload.get("rationale", {}), ensure_ascii=False),
                payload["created_at"],
                payload.get("closed_at"),
            ),
        )

    def list_closed_trades(self, limit: int = 20) -> list[dict[str, Any]]:
        """최근 종료된 거래 목록 (최신순)."""

        return self.fetchall(
            "SELECT * FROM trades WHERE status = 'closed' ORDER BY closed_at DESC LIMIT ?",
            (limit,),
        )

    def save_symbol_scores(self, rows: list[dict[str, Any]]) -> None:
        """Persist symbol ranking output."""

        query = """
            INSERT INTO symbol_scores (
                symbol, product_type, total_score, liquidity_score, volatility_score,
                spread_score, depth_score, trend_score, anomaly_score, age_score,
                tradability_score, metrics_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = [
            (
                row["symbol"],
                row["product_type"],
                row["total_score"],
                row.get("liquidity_score"),
                row.get("volatility_score"),
                row.get("spread_score"),
                row.get("depth_score"),
                row.get("trend_score"),
                row.get("anomaly_score"),
                row.get("age_score"),
                row.get("tradability_score"),
                json.dumps(row.get("metrics", {}), ensure_ascii=False),
                row["created_at"],
            )
            for row in rows
        ]
        self.executemany(query, params)

    def save_news_item(self, payload: dict[str, Any]) -> None:
        """Persist a raw news item."""

        self.execute(
            """
            INSERT OR IGNORE INTO news_items (
                news_hash, source, title, url, published_at, content,
                related_assets_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["news_hash"],
                payload["source"],
                payload["title"],
                payload.get("url"),
                payload.get("published_at"),
                payload.get("content"),
                json.dumps(payload.get("related_assets", []), ensure_ascii=False),
                payload["created_at"],
            ),
        )

    def save_ai_news_analysis(self, payload: dict[str, Any]) -> None:
        """Persist AI-structured news analysis."""

        self.execute(
            """
            INSERT OR REPLACE INTO ai_news_analysis (
                news_hash, summary_ko, impacted_assets_json, impact_level, direction_bias,
                validity_window_minutes, confidence, event_type, should_block_new_entries,
                notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["news_hash"],
                payload["summary_ko"],
                json.dumps(payload.get("impacted_assets", []), ensure_ascii=False),
                payload.get("impact_level"),
                payload.get("direction_bias"),
                payload.get("validity_window_minutes"),
                payload.get("confidence"),
                payload.get("event_type"),
                int(payload.get("should_block_new_entries", False)),
                payload.get("notes"),
                payload["created_at"],
            ),
        )

    def save_daily_stats(self, payload: dict[str, Any]) -> None:
        """Persist daily performance statistics."""

        self.execute(
            """
            INSERT INTO daily_stats (
                trade_date, mode, realized_pnl, unrealized_pnl, pnl_r, fees, trade_count,
                win_count, loss_count, max_drawdown, metadata_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(trade_date) DO UPDATE SET
                mode=excluded.mode,
                realized_pnl=excluded.realized_pnl,
                unrealized_pnl=excluded.unrealized_pnl,
                pnl_r=excluded.pnl_r,
                fees=excluded.fees,
                trade_count=excluded.trade_count,
                win_count=excluded.win_count,
                loss_count=excluded.loss_count,
                max_drawdown=excluded.max_drawdown,
                metadata_json=excluded.metadata_json,
                updated_at=excluded.updated_at
            """,
            (
                payload["trade_date"],
                payload["mode"],
                payload.get("realized_pnl", 0.0),
                payload.get("unrealized_pnl", 0.0),
                payload.get("pnl_r", 0.0),
                payload.get("fees", 0.0),
                payload.get("trade_count", 0),
                payload.get("win_count", 0),
                payload.get("loss_count", 0),
                payload.get("max_drawdown", 0.0),
                json.dumps(payload.get("metadata", {}), ensure_ascii=False),
                payload["updated_at"],
            ),
        )

    def snapshot_settings(self, mode: str, payload: dict[str, Any], created_at: str) -> None:
        """Store settings change history."""

        self.execute(
            "INSERT INTO settings_snapshots (mode, payload_json, created_at) VALUES (?, ?, ?)",
            (mode, json.dumps(payload, ensure_ascii=False), created_at),
        )
