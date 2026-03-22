"""Partial TP, break-even, and local protective logic."""

from __future__ import annotations

from datetime import UTC, datetime
from dataclasses import dataclass
from typing import Any

from core.enums import Side
from core.logger import get_logger
from core.settings import AppSettings
from risk.risk_engine import ApprovedTrade


@dataclass(slots=True)
class ManagedTrade:
    """Runtime-managed exit state."""

    symbol: str
    side: Side
    entry_price: float
    stop_price: float
    tp1_price: float
    tp2_price: float
    tp3_price: float | None
    final_target_price: float | None
    quantity: float
    remaining_quantity: float
    initial_risk: float
    opened_at: datetime
    stop_reason: str = ""
    target_plan: list[dict[str, Any]] | None = None
    tp1_done: bool = False
    tp2_done: bool = False
    tp3_done: bool = False
    break_even_moved: bool = False


class SLTPManager:
    """Track partial TP and break-even logic."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._managed: dict[str, ManagedTrade] = {}
        self.logger = get_logger(__name__)

    def register(self, approved_trade: ApprovedTrade, *, actual_fill_price: float | None = None) -> ManagedTrade:
        """Register a filled entry for exit management.

        actual_fill_price: 실제 체결가 (시장가 fallback 등으로 신호 entry와 다를 수 있음).
        체결가가 이미 TP를 넘었으면 해당 TP를 skip 처리한다.
        """

        fill = actual_fill_price or approved_trade.signal.entry_price
        side = approved_trade.signal.side

        tp1 = approved_trade.tp1_price
        tp2 = approved_trade.tp2_price
        tp3 = approved_trade.tp3_price

        # 실제 체결가가 TP를 이미 넘어선 경우 → 해당 TP skip (즉시 청산 방지)
        tp1_done = False
        tp2_done = False
        tp3_done = False
        if side == Side.LONG:
            if tp1 is not None and fill >= tp1:
                tp1_done = True
                self.logger.warning("TP1 already passed at fill price — skipping", extra={"extra_data": {"symbol": approved_trade.signal.symbol, "fill": fill, "tp1": tp1}})
            if tp2 is not None and fill >= tp2:
                tp2_done = True
            if tp3 is not None and fill >= tp3:
                tp3_done = True
        else:
            if tp1 is not None and fill <= tp1:
                tp1_done = True
                self.logger.warning("TP1 already passed at fill price — skipping", extra={"extra_data": {"symbol": approved_trade.signal.symbol, "fill": fill, "tp1": tp1}})
            if tp2 is not None and fill <= tp2:
                tp2_done = True
            if tp3 is not None and fill <= tp3:
                tp3_done = True

        managed = ManagedTrade(
            symbol=approved_trade.signal.symbol,
            side=side,
            entry_price=fill,
            stop_price=approved_trade.stop_price,
            tp1_price=tp1,
            tp2_price=tp2,
            tp3_price=tp3,
            final_target_price=self._extract_final_target_price(approved_trade.target_plan, side, fill),
            quantity=approved_trade.quantity,
            remaining_quantity=approved_trade.quantity,
            initial_risk=max(abs(fill - approved_trade.stop_price), 1e-9),
            opened_at=datetime.now(tz=UTC),
            stop_reason=approved_trade.stop_reason,
            target_plan=approved_trade.target_plan,
            tp1_done=tp1_done,
            tp2_done=tp2_done,
            tp3_done=tp3_done,
        )
        self._managed[managed.symbol] = managed
        return managed

    def register_payload(self, payload: dict[str, Any]) -> ManagedTrade:
        """Register from persisted payload after a fill or restart."""

        side_value = payload.get("side", "long")
        side = Side.LONG if side_value == "long" else Side.SHORT
        fill = float(
            payload.get("avg_fill_price") or payload.get("entry_price") or payload.get("price")
        )
        tp1 = float(payload.get("tp1_price") or 0.0) or None
        tp2 = float(payload.get("tp2_price") or 0.0) or None
        tp3 = float(payload.get("tp3_price") or 0.0) or None

        # 실제 체결가가 TP를 이미 넘어선 경우 → 즉시 청산 방지
        tp1_done = bool(payload.get("tp1_done", False))
        tp2_done = bool(payload.get("tp2_done", False))
        tp3_done = bool(payload.get("tp3_done", False))
        if not tp1_done and tp1 is not None:
            if (side == Side.LONG and fill >= tp1) or (side == Side.SHORT and fill <= tp1):
                tp1_done = True
                self.logger.warning("TP1 already passed at fill — skipping", extra={"extra_data": {"symbol": payload.get("symbol"), "fill": fill, "tp1": tp1}})
        if not tp2_done and tp2 is not None:
            if (side == Side.LONG and fill >= tp2) or (side == Side.SHORT and fill <= tp2):
                tp2_done = True
        if not tp3_done and tp3 is not None:
            if (side == Side.LONG and fill >= tp3) or (side == Side.SHORT and fill <= tp3):
                tp3_done = True

        managed = ManagedTrade(
            symbol=payload["symbol"],
            side=side,
            entry_price=fill,
            stop_price=float(payload.get("stop_price") or 0),
            tp1_price=tp1,
            tp2_price=tp2,
            tp3_price=tp3,
            final_target_price=self._extract_final_target_price(
                payload.get("target_plan") if isinstance(payload.get("target_plan"), list) else [],
                side,
                fill,
            ),
            quantity=float(payload.get("quantity") or payload.get("filled_quantity")),
            remaining_quantity=float(payload.get("remaining_quantity") or payload.get("quantity") or payload.get("filled_quantity")),
            initial_risk=max(abs(fill - float(payload.get("stop_price") or 0)), 1e-9),
            opened_at=datetime.fromisoformat(str(payload.get("opened_at")).replace("Z", "+00:00"))
            if payload.get("opened_at")
            else datetime.now(tz=UTC),
            stop_reason=str(payload.get("stop_reason") or ""),
            target_plan=payload.get("target_plan") if isinstance(payload.get("target_plan"), list) else [],
            tp1_done=tp1_done,
            tp2_done=tp2_done,
            tp3_done=tp3_done,
            break_even_moved=bool(payload.get("break_even_moved", False)),
        )
        self._managed[managed.symbol] = managed
        return managed

    def remove(self, symbol: str) -> None:
        """Remove a managed trade."""

        self._managed.pop(symbol, None)

    def evaluate_price(self, symbol: str, mark_price: float) -> list[dict[str, Any]]:
        """Return actions to execute for a managed trade."""

        trade = self._managed.get(symbol)
        if trade is None:
            return []
        actions: list[dict[str, Any]] = []
        if trade.side == Side.LONG:
            tp1_hit = trade.tp1_price is not None and trade.tp1_price > 0 and mark_price >= trade.tp1_price
            tp2_hit = trade.tp2_price is not None and trade.tp2_price > 0 and mark_price >= trade.tp2_price
            tp3_hit = trade.tp3_price is not None and trade.tp3_price > 0 and mark_price >= trade.tp3_price
            final_target_hit = trade.final_target_price is not None and mark_price >= trade.final_target_price
            stop_hit = trade.stop_price > 0 and mark_price <= trade.stop_price
        else:
            tp1_hit = trade.tp1_price is not None and trade.tp1_price > 0 and mark_price <= trade.tp1_price
            tp2_hit = trade.tp2_price is not None and trade.tp2_price > 0 and mark_price <= trade.tp2_price
            tp3_hit = trade.tp3_price is not None and trade.tp3_price > 0 and mark_price <= trade.tp3_price
            final_target_hit = trade.final_target_price is not None and mark_price <= trade.final_target_price
            stop_hit = trade.stop_price > 0 and mark_price >= trade.stop_price

        milestone_action: str | None = None
        target_closed_pct = 0.0
        if final_target_hit and trade.remaining_quantity > 0:
            milestone_action = "final_target_exit"
            target_closed_pct = 1.0
            trade.tp1_done = True
            trade.tp2_done = True
            trade.tp3_done = True
        elif tp3_hit and trade.remaining_quantity > 0 and not trade.tp3_done:
            milestone_action = "partial_tp3"
            # TP3: 남은 수량의 50% 익절
            close_qty = trade.remaining_quantity * 0.5
            trade.tp3_done = True
        elif tp2_hit and trade.remaining_quantity > 0 and not trade.tp2_done:
            milestone_action = "partial_tp2"
            # TP2: 남은 수량의 50% 익절
            close_qty = trade.remaining_quantity * 0.5
            trade.tp2_done = True
        elif tp1_hit and trade.remaining_quantity > 0 and not trade.tp1_done:
            milestone_action = "partial_tp1"
            # TP1: 남은 수량의 50% 익절
            close_qty = trade.remaining_quantity * 0.5
            trade.tp1_done = True

        if milestone_action is not None and trade.remaining_quantity > 0:
            # 이미 close_qty가 설정됨 (남은 수량의 50%)
            if "close_qty" not in locals():
                desired_remaining = max(0.0, trade.quantity * (1.0 - target_closed_pct))
                close_qty = min(trade.remaining_quantity, max(0.0, trade.remaining_quantity - desired_remaining))
            if milestone_action == "final_target_exit":
                close_qty = trade.remaining_quantity
            if close_qty > 0:
                actions.append({"action": milestone_action, "symbol": symbol, "quantity": close_qty, "pending_fill": True})
            if (
                milestone_action != "final_target_exit"
                and self.settings.risk.move_sl_to_be_after_tp1
                and not trade.break_even_moved
                and trade.remaining_quantity > 0
            ):
                if trade.side == Side.LONG:
                    trade.stop_price = trade.entry_price + trade.initial_risk * self.settings.risk.be_offset_r
                else:
                    trade.stop_price = trade.entry_price - trade.initial_risk * self.settings.risk.be_offset_r
                trade.break_even_moved = True
                actions.append({"action": "move_stop", "symbol": symbol, "new_stop_price": trade.stop_price})

        hold_minutes = (datetime.now(tz=UTC) - trade.opened_at).total_seconds() / 60
        if hold_minutes >= self.settings.risk.max_position_hold_minutes and trade.remaining_quantity > 0:
            actions.append({"action": "time_stop", "symbol": symbol, "quantity": trade.remaining_quantity})
            trade.remaining_quantity = 0.0
        elif stop_hit and trade.remaining_quantity > 0:
            actions.append({"action": "stop_out", "symbol": symbol, "quantity": trade.remaining_quantity})
        return actions

    def confirm_partial_fill(self, symbol: str, filled_quantity: float) -> None:
        """Decrement remaining_quantity only after fill is confirmed by exchange."""

        trade = self._managed.get(symbol)
        if trade is None:
            return
        trade.remaining_quantity = max(0.0, trade.remaining_quantity - filled_quantity)
        self.logger.info("TP fill confirmed", extra={"extra_data": {
            "symbol": symbol, "filled": filled_quantity, "remaining": trade.remaining_quantity,
        }})

    def revert_pending_action(self, symbol: str, action: str) -> None:
        """Revert TP flags if the exit order failed to fill."""

        trade = self._managed.get(symbol)
        if trade is None:
            return
        if action in ("partial_tp1",):
            trade.tp1_done = False
        elif action in ("partial_tp2",):
            trade.tp1_done = False
            trade.tp2_done = False
        elif action in ("partial_tp3",):
            trade.tp1_done = False
            trade.tp2_done = False
            trade.tp3_done = False
        elif action in ("final_target_exit",):
            trade.tp1_done = False
            trade.tp2_done = False
            trade.tp3_done = False
        self.logger.warning("TP action reverted", extra={"extra_data": {"symbol": symbol, "action": action}})

    def _extract_final_target_price(
        self,
        target_plan: list[dict[str, Any]],
        side: Side,
        entry_price: float,
    ) -> float | None:
        """Pick the primary structural target price from the target plan."""

        for item in target_plan:
            if not isinstance(item, dict):
                continue
            raw_price = item.get("price")
            if raw_price in (None, ""):
                continue
            price = float(raw_price)
            if side == Side.LONG and price > entry_price:
                return price
            if side == Side.SHORT and price < entry_price:
                return price
        return None
