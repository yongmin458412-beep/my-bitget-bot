"""Journal persistence helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from core.persistence import SQLitePersistence


class TradeJournal:
    """Handle signals, entries, exits, and /why lookups."""

    def __init__(self, persistence: SQLitePersistence) -> None:
        self.persistence = persistence

    def log_signal(self, payload: dict[str, Any]) -> None:
        """Persist a signal snapshot."""

        self.persistence.save_signal(payload)

    def log_entry(self, payload: dict[str, Any]) -> None:
        """Persist a trade entry into the journal."""

        self.persistence.save_trade(
            {
                "trade_id": payload.get("trade_id") or payload.get("signal_id"),
                "symbol": payload["symbol"],
                "strategy": payload.get("strategy", ""),
                "side": payload["side"],
                "mode": payload["mode"],
                "entry_price": payload.get("avg_fill_price", payload.get("price")),
                "stop_price": payload.get("stop_price"),
                "tp1_price": payload.get("tp1_price"),
                "tp2_price": payload.get("tp2_price"),
                "tp3_price": payload.get("tp3_price"),
                "quantity": payload.get("filled_quantity", payload.get("quantity")),
                "realized_pnl": 0.0,
                "realized_pnl_usdt": 0.0,
                "pnl_r": 0.0,
                "status": "open",
                "stop_reason": payload.get("stop_reason", ""),
                "tp1_reason": self._target_reason(payload, 0),
                "tp2_reason": self._target_reason(payload, 1),
                "tp3_reason": self._target_reason(payload, 2),
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
                "trade_rejected_reason": payload.get("trade_rejected_reason"),
                "ev_metrics": payload.get("ev_metrics", {}),
                "tags": payload.get("tags", []),
                "rationale": payload.get("rationale", {}),
                "created_at": payload.get("created_at", datetime.now(tz=UTC).isoformat(timespec="seconds")),
            }
        )

    def log_exit(self, payload: dict[str, Any]) -> None:
        """Persist a trade close into the journal."""

        existing = self.persistence.fetchone(
            "SELECT * FROM trades WHERE symbol = ? AND status = 'open' ORDER BY created_at DESC LIMIT 1",
            (payload["symbol"],),
        )
        if existing is None:
            return
        self.persistence.save_trade(
            {
                "trade_id": existing["trade_id"],
                "symbol": existing["symbol"],
                "strategy": existing["strategy"],
                "side": existing["side"],
                "mode": existing["mode"],
                "entry_price": existing["entry_price"],
                "exit_price": payload.get("avg_fill_price", payload.get("price")),
                "stop_price": existing["stop_price"],
                "tp1_price": existing["tp1_price"],
                "tp2_price": existing["tp2_price"],
                "tp3_price": existing.get("tp3_price"),
                "quantity": existing["quantity"],
                "realized_pnl": payload.get("realized_pnl", 0.0),
                "realized_pnl_usdt": payload.get("realized_pnl_usdt", payload.get("realized_pnl", 0.0)),
                "pnl_r": payload.get("pnl_r", 0.0),
                "hold_minutes": payload.get("hold_minutes", 0.0),
                "status": "closed",
                "exit_reason": payload.get("exit_reason", "manual"),
                "stop_reason": payload.get("stop_reason", existing.get("stop_reason", "")),
                "tp1_reason": existing.get("tp1_reason"),
                "tp2_reason": existing.get("tp2_reason"),
                "tp3_reason": existing.get("tp3_reason"),
                "rr_to_tp1": payload.get("rr_to_tp1", existing.get("rr_to_tp1", 0.0)),
                "rr_to_tp2": payload.get("rr_to_tp2", existing.get("rr_to_tp2", 0.0)),
                "rr_to_best_target": payload.get("rr_to_best_target", existing.get("rr_to_best_target", 0.0)),
                "market_regime": payload.get("market_regime", existing.get("market_regime", "")),
                "chosen_strategy": payload.get("chosen_strategy", existing.get("chosen_strategy", existing.get("strategy", ""))),
                "candidate_strategies": payload.get("candidate_strategies", self._as_json_list(existing.get("candidate_strategies_json"))),
                "rejected_strategies": payload.get("rejected_strategies", self._as_json_list(existing.get("rejected_strategies_json"))),
                "rejection_reasons": payload.get("rejection_reasons", self._as_json_list(existing.get("rejection_reasons_json"))),
                "overlap_score": payload.get("overlap_score", existing.get("overlap_score", 0.0)),
                "conflict_resolution_decision": payload.get("conflict_resolution_decision", existing.get("conflict_resolution_decision", "")),
                "trade_rejected_reason": payload.get("trade_rejected_reason", existing.get("trade_rejected_reason")),
                "ev_metrics": payload.get("ev_metrics", self._as_json_dict(existing.get("ev_metrics_json"))),
                "tags": payload.get("tags", []),
                "rationale": payload.get("rationale", {}),
                "created_at": existing["created_at"],
                "closed_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            }
        )

    def recent_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent journal rows."""

        return self.persistence.fetchall(
            "SELECT * FROM trades ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )

    def recent_signals(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent signals."""

        return self.persistence.fetchall(
            "SELECT * FROM signals ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )

    def why_symbol(self, symbol: str) -> dict[str, Any]:
        """Return latest signal/trade rationale for a symbol."""

        signal = self.persistence.fetchone(
            "SELECT * FROM signals WHERE symbol = ? ORDER BY created_at DESC LIMIT 1",
            (symbol.upper(),),
        )
        trade = self.persistence.fetchone(
            "SELECT * FROM trades WHERE symbol = ? ORDER BY created_at DESC LIMIT 1",
            (symbol.upper(),),
        )
        return {
            "symbol": symbol.upper(),
            "signal": signal,
            "trade": trade,
            "rule_summary": self._build_rule_summary(signal, trade),
        }

    def _as_json_list(self, value: Any) -> list[Any]:
        """Return a list from a DB JSON string or python value."""

        if isinstance(value, list):
            return value
        if isinstance(value, str) and value:
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return []
            return parsed if isinstance(parsed, list) else []
        return []

    def _as_json_dict(self, value: Any) -> dict[str, Any]:
        """Return a dict from a DB JSON string or python value."""

        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value:
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    def _target_reason(self, payload: dict[str, Any], index: int) -> str:
        """Extract target reason from target plan."""

        plan = payload.get("target_plan") or []
        if not isinstance(plan, list) or len(plan) <= index or not isinstance(plan[index], dict):
            return ""
        return str(plan[index].get("reason") or "")

    def _build_rule_summary(self, signal: dict[str, Any] | None, trade: dict[str, Any] | None) -> dict[str, Any]:
        """Build a human-readable structural explanation payload."""

        signal = signal or {}
        trade = trade or {}
        target_plan = self._as_json_list(signal.get("target_plan_json")) or self._as_json_list(trade.get("target_plan_json"))
        stop_reason = str(trade.get("stop_reason") or signal.get("stop_reason") or "-")
        target_reasons = [str(item.get("reason", "")) for item in target_plan[:3] if isinstance(item, dict)]
        if not target_reasons:
            target_reasons = [
                str(value)
                for value in [trade.get("tp1_reason"), trade.get("tp2_reason"), trade.get("tp3_reason")]
                if str(value or "").strip()
            ]
        rr_to_tp1 = trade.get("rr_to_tp1") or signal.get("rr_to_tp1") or 0
        rr_to_tp2 = trade.get("rr_to_tp2") or signal.get("rr_to_tp2") or 0
        chosen_strategy = trade.get("chosen_strategy") or signal.get("chosen_strategy") or trade.get("strategy") or signal.get("strategy") or "-"
        market_regime = trade.get("market_regime") or signal.get("market_regime") or "-"
        conflict_decision = trade.get("conflict_resolution_decision") or signal.get("conflict_resolution_decision") or "-"
        rejected_reason = signal.get("trade_rejected_reason") or trade.get("trade_rejected_reason")
        return {
            "summary_ko": (
                f"전략은 '{chosen_strategy}' 를 선택했고 레짐은 '{market_regime}' 로 해석했습니다. "
                f"손절은 '{stop_reason}' 구조가 깨지는 지점에 두고, "
                f"목표가는 {', '.join(target_reasons[:3]) or '가까운 구조 레벨'} 기준으로 배치했습니다."
            ),
            "bullet_points": [
                f"선택 전략: {chosen_strategy}",
                f"시장 레짐: {market_regime}",
                f"손절 사유: {stop_reason}",
                f"목표 사유: {', '.join(target_reasons[:3]) or '-'}",
                f"RR(TP1/TP2): {rr_to_tp1} / {rr_to_tp2}",
                f"충돌 해결: {conflict_decision}",
                f"거절 사유: {rejected_reason or '없음'}",
            ],
            "stop_commentary": f"시나리오가 틀리는 구조 무효화 지점은 {stop_reason} 입니다.",
            "target_commentary": f"목표가는 다음 구조 레벨 {', '.join(target_reasons[:3]) or '-'} 순서로 선택했습니다.",
            "rr_commentary": (
                f"TP1 RR {rr_to_tp1}, TP2 RR {rr_to_tp2} 기준으로 진입 적합성을 평가했습니다."
                + (f" 현재 거절 사유는 {rejected_reason} 입니다." if rejected_reason else "")
            ),
            "target_reasons": target_reasons,
        }
