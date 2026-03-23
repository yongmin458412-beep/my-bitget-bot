"""Risk-reward evaluation for structural trade plans."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from core.enums import Side
from strategy.base import SignalContext


@dataclass(slots=True)
class TradeViabilityResult:
    """Result of structural RR and cost evaluation."""

    rr_to_tp1: float
    rr_to_tp2: float
    rr_to_best_target: float
    fees_impact: float
    slippage_impact: float
    expected_value_proxy: float
    approved: bool
    reject_reason: str = ""
    preferred_rr_to_tp2_met: bool = False
    target_plan: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready version of the result."""

        return asdict(self)


def compute_structural_rr(entry_price: float, stop_price: float, target_price: float | None, side: Side) -> float:
    """Compute RR using a structural stop and a structural target."""

    if target_price is None:
        return 0.0
    if side == Side.LONG:
        risk = entry_price - stop_price
        reward = target_price - entry_price
    else:
        risk = stop_price - entry_price
        reward = entry_price - target_price
    if risk <= 0 or reward <= 0:
        return 0.0
    return reward / risk


def compute_quadrant_targets(entry_price: float, stop_price: float, side: Side) -> dict[str, float]:
    """
    손익비 1:2 — 최종 TP를 SL 거리의 2배로 잡고 4등분.

    SL 거리 = risk
    최종 목표 = risk × 2  (1:2 손익비)
    TP1 = 0.5R (목표의 25%)  — 남은 수량 50% 익절
    TP2 = 1.0R (목표의 50%)  — 남은 수량 50% 익절
    TP3 = 1.5R (목표의 75%)  — 남은 수량 50% 익절
    TP4 = 2.0R (목표의 100%) — 전량청산
    """
    if side == Side.LONG:
        risk = entry_price - stop_price
    else:
        risk = stop_price - entry_price

    if risk <= 0:
        return {}

    # 1:2 손익비: 최종 목표 = risk × 2, 이걸 4등분
    if side == Side.LONG:
        tp1 = entry_price + risk * 0.5   # 0.5R
        tp2 = entry_price + risk * 1.0   # 1.0R
        tp3 = entry_price + risk * 1.5   # 1.5R
        tp4 = entry_price + risk * 2.0   # 2.0R (최종)
    else:
        tp1 = entry_price - risk * 0.5   # 0.5R
        tp2 = entry_price - risk * 1.0   # 1.0R
        tp3 = entry_price - risk * 1.5   # 1.5R
        tp4 = entry_price - risk * 2.0   # 2.0R (최종)

    return {
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "tp4": tp4,
    }


def evaluate_trade_viability(
    signal_context: SignalContext,
    stop_price: float,
    targets: list[dict[str, Any]],
    fees_r: float,
    slippage_r: float,
    *,
    entry_price: float,
    side: Side,
    strategy_name: str,
    min_stop_distance_pct: float = 0.005,
    min_rr_to_tp1_break_retest: float,
    preferred_rr_to_tp2_break_retest: float,
    min_rr_to_tp1_liquidity_raid: float,
    preferred_rr_to_tp2_liquidity_raid: float,
    min_rr_to_tp1_fair_value_gap: float,
    preferred_rr_to_tp2_fair_value_gap: float,
    min_rr_to_tp1_order_block: float,
    preferred_rr_to_tp2_order_block: float,
    min_rr_to_tp1_choch: float,
    preferred_rr_to_tp2_choch: float,
    reject_trade_if_targets_are_inside_range_middle: bool = True,
) -> TradeViabilityResult:
    """Evaluate structural targets as a gating filter rather than a target generator."""

    # SL이 최소 거리보다 가까우면 거래 거부
    if stop_price > 0:
        if side == Side.LONG:
            risk_pct = (entry_price - stop_price) / entry_price
        else:
            risk_pct = (stop_price - entry_price) / entry_price
        if risk_pct < min_stop_distance_pct:
            return TradeViabilityResult(
                rr_to_tp1=0.0,
                rr_to_tp2=0.0,
                rr_to_best_target=0.0,
                fees_impact=fees_r,
                slippage_impact=slippage_r,
                expected_value_proxy=-(fees_r + slippage_r),
                approved=False,
                reject_reason=f"stop_too_close (risk_pct={risk_pct:.4f}<{min_stop_distance_pct:.4f})",
                target_plan=[],
            )

    if not targets:
        return TradeViabilityResult(
            rr_to_tp1=0.0,
            rr_to_tp2=0.0,
            rr_to_best_target=0.0,
            fees_impact=fees_r,
            slippage_impact=slippage_r,
            expected_value_proxy=-(fees_r + slippage_r),
            approved=False,
            reject_reason="insufficient_structural_targets",
            target_plan=[],
        )

    rr_values = [compute_structural_rr(entry_price, stop_price, target["price"], side) for target in targets]
    rr_to_tp1 = rr_values[0] if rr_values else 0.0
    rr_to_tp2 = rr_values[1] if len(rr_values) > 1 else 0.0
    rr_to_best_target = max(rr_values) if rr_values else 0.0

    if strategy_name == "liquidity_raid":
        min_rr_to_tp1 = min_rr_to_tp1_liquidity_raid
        preferred_rr_to_tp2 = preferred_rr_to_tp2_liquidity_raid
    elif strategy_name == "fair_value_gap":
        min_rr_to_tp1 = min_rr_to_tp1_fair_value_gap
        preferred_rr_to_tp2 = preferred_rr_to_tp2_fair_value_gap
    elif strategy_name == "order_block":
        min_rr_to_tp1 = min_rr_to_tp1_order_block
        preferred_rr_to_tp2 = preferred_rr_to_tp2_order_block
    elif strategy_name == "choch":
        min_rr_to_tp1 = min_rr_to_tp1_choch
        preferred_rr_to_tp2 = preferred_rr_to_tp2_choch
    else:
        min_rr_to_tp1 = min_rr_to_tp1_break_retest
        preferred_rr_to_tp2 = preferred_rr_to_tp2_break_retest

    if reject_trade_if_targets_are_inside_range_middle and targets and bool(targets[0].get("inside_range_middle")):
        return TradeViabilityResult(
            rr_to_tp1=rr_to_tp1,
            rr_to_tp2=rr_to_tp2,
            rr_to_best_target=rr_to_best_target,
            fees_impact=fees_r,
            slippage_impact=slippage_r,
            expected_value_proxy=rr_to_best_target - fees_r - slippage_r,
            approved=False,
            reject_reason="tp_inside_range_middle",
            target_plan=targets,
        )

    spread_penalty = max(0.0, float(signal_context.ticker.get("spread_bps", 0.0)) / 10_000)
    event_penalty = float(signal_context.news_penalty)
    net_rr_to_tp1 = rr_to_tp1 - fees_r - slippage_r - spread_penalty - event_penalty
    if rr_to_tp1 < min_rr_to_tp1:
        return TradeViabilityResult(
            rr_to_tp1=rr_to_tp1,
            rr_to_tp2=rr_to_tp2,
            rr_to_best_target=rr_to_best_target,
            fees_impact=fees_r,
            slippage_impact=slippage_r,
            expected_value_proxy=net_rr_to_tp1,
            approved=False,
            reject_reason="min_rr_to_tp1_failed",
            target_plan=targets,
        )
    if net_rr_to_tp1 <= 0:
        return TradeViabilityResult(
            rr_to_tp1=rr_to_tp1,
            rr_to_tp2=rr_to_tp2,
            rr_to_best_target=rr_to_best_target,
            fees_impact=fees_r,
            slippage_impact=slippage_r,
            expected_value_proxy=net_rr_to_tp1,
            approved=False,
            reject_reason="fees_slippage_dominate_tp1",
            target_plan=targets,
        )

    expected_value_proxy = (rr_to_tp1 * 0.55) + (rr_to_best_target * 0.45) - fees_r - slippage_r - spread_penalty - event_penalty
    return TradeViabilityResult(
        rr_to_tp1=rr_to_tp1,
        rr_to_tp2=rr_to_tp2,
        rr_to_best_target=rr_to_best_target,
        fees_impact=fees_r,
        slippage_impact=slippage_r,
        expected_value_proxy=expected_value_proxy,
        approved=True,
        preferred_rr_to_tp2_met=rr_to_tp2 >= preferred_rr_to_tp2,
        target_plan=targets,
    )
