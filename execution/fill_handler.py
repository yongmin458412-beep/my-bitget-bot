"""Fill processing and journal/persistence updates."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from core.logger import get_logger
from core.persistence import SQLitePersistence
from core.state_store import StateStore
from journal.trade_journal import TradeJournal


class FillHandler:
    """Update order, position, and trade records from fills."""

    def __init__(
        self,
        persistence: SQLitePersistence,
        state_store: StateStore,
        trade_journal: TradeJournal,
    ) -> None:
        self.persistence = persistence
        self.state_store = state_store
        self.trade_journal = trade_journal
        self.logger = get_logger(__name__)

    def _verify_fill(self, payload: dict[str, Any]) -> bool:
        """Validate fill data consistency before persisting."""

        filled_qty = float(payload.get("filled_quantity") or payload.get("quantity") or 0)
        requested_qty = float(payload.get("quantity") or 0)
        avg_price = float(payload.get("avg_fill_price") or payload.get("price") or 0)

        if filled_qty <= 0:
            self.logger.error("Fill rejected: zero/negative quantity", extra={"extra_data": {
                "symbol": payload.get("symbol"), "filled_quantity": filled_qty,
            }})
            return False
        if avg_price <= 0:
            self.logger.error("Fill rejected: zero/negative fill price", extra={"extra_data": {
                "symbol": payload.get("symbol"), "avg_fill_price": avg_price,
            }})
            return False
        if requested_qty > 0 and abs(filled_qty - requested_qty) / requested_qty > 0.01:
            self.logger.warning("Fill quantity mismatch", extra={"extra_data": {
                "symbol": payload.get("symbol"),
                "requested": requested_qty, "filled": filled_qty,
                "diff_pct": round(abs(filled_qty - requested_qty) / requested_qty * 100, 2),
            }})
        return True

    def on_entry_filled(self, payload: dict[str, Any]) -> None:
        """Persist an entry fill after verification."""

        if not self._verify_fill(payload):
            return

        now_iso = datetime.now(tz=UTC).isoformat(timespec="seconds")
        self.persistence.save_order(
            {
                "client_order_id": payload["client_order_id"],
                "exchange_order_id": payload.get("exchange_order_id"),
                "symbol": payload["symbol"],
                "mode": payload["mode"],
                "side": payload["side"],
                "order_type": payload["order_type"],
                "status": "filled",
                "price": payload.get("price"),
                "quantity": payload.get("quantity"),
                "filled_quantity": payload.get("filled_quantity", payload.get("quantity")),
                "avg_fill_price": payload.get("avg_fill_price", payload.get("price")),
                "reduce_only": False,
                "reason": payload.get("reason"),
                "created_at": payload.get("created_at", now_iso),
                "updated_at": now_iso,
            }
        )
        self.persistence.save_position(
            {
                "symbol": payload["symbol"],
                "mode": payload["mode"],
                "side": payload["side"],
                "status": "open",
                "entry_price": payload.get("avg_fill_price", payload.get("price")),
                "mark_price": payload.get("avg_fill_price", payload.get("price")),
                "stop_price": payload.get("stop_price"),
                "tp1_price": payload.get("tp1_price"),
                "tp2_price": payload.get("tp2_price"),
                "tp3_price": payload.get("tp3_price"),
                "quantity": payload.get("filled_quantity", payload.get("quantity")),
                "leverage": payload.get("leverage"),
                "used_margin": payload.get("used_margin"),
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
                "strategy": payload.get("strategy"),
                "signal_id": payload.get("signal_id"),
                "metadata": {
                    **(payload.get("metadata", {}) or {}),
                    "display_name": payload.get("display_name", ""),
                    "rationale": payload.get("rationale", {}),
                    "tags": payload.get("tags", []),
                },
                "stop_reason": payload.get("stop_reason", ""),
                "target_plan": payload.get("target_plan", []),
                "rr_to_tp1": payload.get("rr_to_tp1", 0.0),
                "rr_to_tp2": payload.get("rr_to_tp2", 0.0),
                "rr_to_best_target": payload.get("rr_to_best_target", 0.0),
                "market_regime": payload.get("market_regime", ""),
                "chosen_strategy": payload.get("chosen_strategy", payload.get("strategy", "")),
                "candidate_strategies": payload.get("candidate_strategies", []),
                "rejected_strategies": payload.get("rejected_strategies", []),
                "rejection_reasons": payload.get("rejection_reasons", []),
                "overlap_score": payload.get("overlap_score", 0.0),
                "conflict_resolution_decision": payload.get("conflict_resolution_decision", ""),
                "updated_at": now_iso,
            }
        )
        self.state_store.update_position(payload["symbol"], payload)
        self.trade_journal.log_entry(payload)

    def on_position_closed(self, payload: dict[str, Any]) -> None:
        """Persist a closing fill."""

        now_iso = datetime.now(tz=UTC).isoformat(timespec="seconds")
        try:
            self.persistence.close_position(payload["symbol"], payload["mode"], payload["side"], now_iso)
        except Exception:
            # 이미 closed 상태이거나 UNIQUE constraint — 무시하고 state 정리는 계속 진행
            pass
        try:
            self.trade_journal.log_exit(payload)
        except Exception:
            pass
        self.state_store.remove_position(payload["symbol"])
