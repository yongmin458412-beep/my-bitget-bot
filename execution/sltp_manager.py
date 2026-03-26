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
    tp4_price: float | None
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
    tp4_done: bool = False
    break_even_moved: bool = False
    # 트레일링 스탑 필드
    trailing_active: bool = False
    trailing_extreme: float = 0.0   # 롱: 최고가 / 숏: 최저가 추적
    trailing_atr: float = 0.0       # 트레일링 거리 (= initial_risk)


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
        initial_risk = max(abs(fill - approved_trade.stop_price), 1e-9)

        # TP1 = 1R (진입가 + 리스크 거리), TP2~4는 트레일링으로 대체
        if side == Side.LONG:
            tp1 = fill + initial_risk
        else:
            tp1 = fill - initial_risk
        tp2 = approved_trade.tp2_price
        tp3 = approved_trade.tp3_price
        tp4 = approved_trade.tp4_price

        # 실제 체결가가 TP를 이미 넘어선 경우 → 해당 TP skip (즉시 청산 방지)
        tp1_done = False
        tp2_done = False
        tp3_done = False
        tp4_done = False
        if side == Side.LONG:
            if tp1 is not None and fill >= tp1:
                tp1_done = True
                self.logger.warning("TP1 already passed at fill price — skipping", extra={"extra_data": {"symbol": approved_trade.signal.symbol, "fill": fill, "tp1": tp1}})
            if tp2 is not None and fill >= tp2:
                tp2_done = True
            if tp3 is not None and fill >= tp3:
                tp3_done = True
            if tp4 is not None and fill >= tp4:
                tp4_done = True
        else:
            if tp1 is not None and fill <= tp1:
                tp1_done = True
                self.logger.warning("TP1 already passed at fill price — skipping", extra={"extra_data": {"symbol": approved_trade.signal.symbol, "fill": fill, "tp1": tp1}})
            if tp2 is not None and fill <= tp2:
                tp2_done = True
            if tp3 is not None and fill <= tp3:
                tp3_done = True
            if tp4 is not None and fill <= tp4:
                tp4_done = True

        managed = ManagedTrade(
            symbol=approved_trade.signal.symbol,
            side=side,
            entry_price=fill,
            stop_price=approved_trade.stop_price,
            tp1_price=tp1,
            tp2_price=tp2,
            tp3_price=tp3,
            tp4_price=tp4,
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
            tp4_done=tp4_done,
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
        stop_price = float(payload.get("stop_price") or 0)
        initial_risk = max(abs(fill - stop_price), 1e-9)

        # TP1 = 1R (진입가 + 리스크 거리), TP2~4는 트레일링으로 대체
        if side == Side.LONG:
            tp1 = fill + initial_risk
        else:
            tp1 = fill - initial_risk
        tp2 = float(payload.get("tp2_price") or 0.0) or None
        tp3 = float(payload.get("tp3_price") or 0.0) or None
        tp4 = float(payload.get("tp4_price") or 0.0) or None

        # 실제 체결가가 TP를 이미 넘어선 경우 → 즉시 청산 방지
        tp1_done = bool(payload.get("tp1_done", False))
        tp2_done = bool(payload.get("tp2_done", False))
        tp3_done = bool(payload.get("tp3_done", False))
        tp4_done = bool(payload.get("tp4_done", False))
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
        if not tp4_done and tp4 is not None:
            if (side == Side.LONG and fill >= tp4) or (side == Side.SHORT and fill <= tp4):
                tp4_done = True

        managed = ManagedTrade(
            symbol=payload["symbol"],
            side=side,
            entry_price=fill,
            stop_price=float(payload.get("stop_price") or 0),
            tp1_price=tp1,
            tp2_price=tp2,
            tp3_price=tp3,
            tp4_price=tp4,
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
            tp4_done=tp4_done,
            break_even_moved=bool(payload.get("break_even_moved", False)),
        )
        self._managed[managed.symbol] = managed
        return managed

    def remove(self, symbol: str) -> None:
        """Remove a managed trade."""

        self._managed.pop(symbol, None)

    def evaluate_price(self, symbol: str, mark_price: float, *, open_position_count: int = 1) -> list[dict[str, Any]]:
        """Return actions to execute for a managed trade.

        TP 전략: 1R 본전 + ATR 트레일링
          - TP1: 1R 도달 시 50% 청산 + SL → 본절 이동
          - 나머지 50%: ATR 트레일링 스탑으로 추세 추적
        """

        trade = self._managed.get(symbol)
        if trade is None:
            return []
        actions: list[dict[str, Any]] = []
        is_long = trade.side == Side.LONG

        # ── 1) TP1 (1R) 체크: 50% 청산 ─────────────────────────────────────
        tp1_hit = False
        if trade.tp1_price is not None and trade.tp1_price > 0:
            tp1_hit = mark_price >= trade.tp1_price if is_long else mark_price <= trade.tp1_price

        stop_hit = False
        if trade.stop_price > 0:
            stop_hit = mark_price <= trade.stop_price if is_long else mark_price >= trade.stop_price

        if tp1_hit and trade.remaining_quantity > 0 and not trade.tp1_done:
            half = trade.quantity * 0.50
            close_qty = min(half, trade.remaining_quantity)
            actions.append({"action": "partial_tp1", "symbol": symbol, "quantity": close_qty, "pending_fill": True})
            trade.tp1_done = True

            # SL → 본절 이동
            if not trade.break_even_moved:
                be_offset = trade.initial_risk * self.settings.risk.be_offset_r
                if is_long:
                    trade.stop_price = trade.entry_price + be_offset
                else:
                    trade.stop_price = trade.entry_price - be_offset
                trade.break_even_moved = True
                actions.append({"action": "move_stop", "symbol": symbol, "new_stop_price": trade.stop_price})

            # 트레일링 스탑 활성화
            trade.trailing_active = True
            trade.trailing_atr = trade.initial_risk
            trade.trailing_extreme = mark_price

        # ── 2) 트레일링 스탑 추적 ──────────────────────────────────────────
        if trade.trailing_active and trade.remaining_quantity > 0:
            # 최고가/최저가 갱신
            if is_long:
                if mark_price > trade.trailing_extreme:
                    trade.trailing_extreme = mark_price
                trailing_stop = trade.trailing_extreme - trade.trailing_atr
                trailing_hit = mark_price <= trailing_stop
            else:
                if mark_price < trade.trailing_extreme:
                    trade.trailing_extreme = mark_price
                trailing_stop = trade.trailing_extreme + trade.trailing_atr
                trailing_hit = mark_price >= trailing_stop

            if trailing_hit:
                actions.append({"action": "trailing_stop", "symbol": symbol, "quantity": trade.remaining_quantity})
                trade.remaining_quantity = 0.0
                return actions

        # ── 3) 보유시간 초과 (3개 이상 포지션일 때만) ────────────────────────
        hold_minutes = (datetime.now(tz=UTC) - trade.opened_at).total_seconds() / 60
        min_positions_for_time_stop = 3
        if (
            hold_minutes >= self.settings.risk.max_position_hold_minutes
            and trade.remaining_quantity > 0
            and open_position_count >= min_positions_for_time_stop
        ):
            actions.append({"action": "time_stop", "symbol": symbol, "quantity": trade.remaining_quantity})
            trade.remaining_quantity = 0.0

        # ── 4) 손절 ────────────────────────────────────────────────────────
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
            trade.tp2_done = False
        elif action in ("partial_tp3",):
            trade.tp3_done = False
        elif action in ("partial_tp4",):
            trade.tp4_done = False
        elif action in ("final_target_exit",):
            trade.tp1_done = False
            trade.tp2_done = False
            trade.tp3_done = False
            trade.tp4_done = False
        self.logger.warning("TP action reverted", extra={"extra_data": {"symbol": symbol, "action": action}})

    def _extract_final_target_price(
        self,
        target_plan: list[dict[str, Any]],
        side: Side,
        entry_price: float,
    ) -> float | None:
        """Pick the furthest target (TP4 = 2.0R) from the target plan."""

        result: float | None = None
        for item in target_plan:
            if not isinstance(item, dict):
                continue
            raw_price = item.get("price")
            if raw_price in (None, ""):
                continue
            price = float(raw_price)
            if side == Side.LONG and price > entry_price:
                if result is None or price > result:
                    result = price  # 가장 먼 타겟 (최고가)
            if side == Side.SHORT and price < entry_price:
                if result is None or price < result:
                    result = price  # 가장 먼 타겟 (최저가)
        return result
