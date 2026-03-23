"""Translate a strategy signal into an approved trade plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from core.enums import Side
from core.settings import AppSettings
from exchange.bitget_models import ContractConfig
from strategy.base import StrategySignal

from .position_sizing import PositionSizer
from .stops import choose_safer_stop
from .trade_guard import GuardCheckResult, TradeGuard


@dataclass(slots=True)
class ApprovedTrade:
    """Trade plan after risk approval."""

    signal: StrategySignal
    quantity: float
    leverage: float
    stop_price: float
    tp1_price: float
    tp2_price: float
    tp3_price: float | None
    risk_amount: float
    stop_reason: str = ""
    target_plan: list[dict[str, Any]] = field(default_factory=list)
    rr_to_tp1: float = 0.0
    rr_to_tp2: float = 0.0
    rr_to_best_target: float = 0.0
    trade_rejected_reason: str | None = None
    ev_metrics: dict[str, Any] = field(default_factory=dict)
    guard_result: GuardCheckResult = field(default_factory=GuardCheckResult)
    notes: list[str] = field(default_factory=list)


class RiskEngine:
    """Validate risk and convert signals into executable trades."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.trade_guard = TradeGuard(settings.risk)
        self.position_sizer = PositionSizer()

    def approve(
        self,
        *,
        signal: StrategySignal,
        contract: ContractConfig,
        account_equity: float,
        open_positions: dict[str, dict[str, Any]],
        runtime_metrics: dict[str, Any],
        atr_value: float,
        news_blocked: bool,
        funding_blocked: bool,
    ) -> ApprovedTrade | None:
        """Return an approved trade or None if blocked."""

        if signal.trade_rejected_reason:
            signal.blockers.append(signal.trade_rejected_reason)
            return None

        # 현재 포트폴리오 히트 계산: 오픈 포지션 증거금 합 / 총자산
        current_portfolio_heat_pct = self._calc_portfolio_heat(open_positions, account_equity)

        guard = self.trade_guard.evaluate(
            symbol=signal.symbol,
            open_positions_count=len(open_positions),
            symbol_has_position=signal.symbol in open_positions,
            daily_loss_r=float(runtime_metrics.get("daily_loss_r", 0.0)),
            consecutive_losses=int(runtime_metrics.get("consecutive_losses", 0)),
            daily_order_count=int(runtime_metrics.get("daily_order_count", 0)),
            paused=bool(runtime_metrics.get("paused", False)),
            news_blocked=news_blocked,
            funding_blocked=funding_blocked,
            account_drawdown_pct=float(runtime_metrics.get("account_drawdown_pct", 0.0)),
            unrealized_loss_pct=float(runtime_metrics.get("unrealized_loss_pct", 0.0)),
            symbol_cooldown_until=runtime_metrics.get("symbol_cooldown_until"),
            current_portfolio_heat_pct=current_portfolio_heat_pct,
        )
        if not guard.approved:
            signal.blockers.extend(guard.blockers)
            return None

        configured_leverage = self.settings.execution.per_strategy_leverage.get(
            signal.strategy.value,
            self.settings.execution.default_leverage,
        )
        leverage = configured_leverage
        if contract.max_leverage is not None:
            leverage = min(leverage, contract.max_leverage)
        if contract.min_leverage is not None:
            leverage = max(leverage, contract.min_leverage)
        stop_price = choose_safer_stop(
            entry_price=signal.entry_price,
            structural_stop=signal.stop_price,
            side=signal.side,
            atr_value=atr_value,
            stop_mode=self.settings.risk.stop_mode,
            use_atr_buffer=self.settings.risk.use_atr_buffer_for_stop,
            atr_buffer_multiplier=self.settings.risk.atr_buffer_multiplier,
            hard_stop_pct=self.settings.risk.hard_stop_loss_pct,
            structural_stop_already_buffered=True,
        )
        if self.settings.risk.sizing_mode == "equity_share":
            sizing = self.position_sizer.size_from_equity_share(
                account_equity=account_equity,
                equity_share_count=self.settings.risk.equity_share_count,
                entry_price=signal.entry_price,
                leverage=leverage,
                contract=contract,
            )
        else:
            sizing = self.position_sizer.size_from_risk(
                account_equity=account_equity,
                risk_fraction=self.settings.risk.risk_per_trade,
                entry_price=signal.entry_price,
                stop_price=stop_price,
                leverage=leverage,
                contract=contract,
            )
        if not sizing.valid:
            signal.blockers.append(sizing.reason)
            return None

        execution_target_plan, tp1_price, tp2_price, tp3_price = self._build_execution_targets(signal)

        return ApprovedTrade(
            signal=signal,
            quantity=sizing.quantity,
            leverage=leverage,
            stop_price=stop_price,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            tp3_price=tp3_price,
            risk_amount=sizing.risk_amount,
            stop_reason=signal.stop_reason,
            target_plan=execution_target_plan,
            rr_to_tp1=signal.rr_to_tp1,
            rr_to_tp2=signal.rr_to_tp2,
            rr_to_best_target=signal.rr_to_best_target,
            trade_rejected_reason=signal.trade_rejected_reason,
            ev_metrics=signal.ev_metrics,
            guard_result=guard,
            notes=[
                f"risk_per_trade={self.settings.risk.risk_per_trade}",
                f"configured_leverage={configured_leverage}",
                f"applied_leverage={leverage}",
                f"contract_max_leverage={contract.max_leverage}",
                f"calculated_at={datetime.now(tz=UTC).isoformat(timespec='seconds')}",
            ],
        )

    def _calc_portfolio_heat(
        self,
        open_positions: dict[str, dict[str, Any]],
        account_equity: float,
    ) -> float:
        """현재 오픈 포지션의 총 증거금 비율 계산 (증거금 합 / 총자산)."""

        if account_equity <= 0:
            return 1.0
        total_margin = 0.0
        for pos in open_positions.values():
            notional = float(pos.get("notional", 0.0) or pos.get("size", 0.0))
            leverage = float(pos.get("leverage", 1.0) or 1.0)
            if leverage > 0:
                total_margin += notional / leverage
        return total_margin / account_equity

    def _build_execution_targets(
        self,
        signal: StrategySignal,
    ) -> tuple[list[dict[str, Any]], float, float, float | None]:
        """Build 1:2 RR scale-out levels (TP1=0.5R, TP2=1.0R, TP3=1.5R) from stop_price."""

        raw_plan = [dict(item) for item in signal.target_plan if isinstance(item, dict)]
        primary = self._pick_primary_target(signal, raw_plan)
        reordered_plan = raw_plan if primary is None else ([primary] + [item for item in raw_plan if item != primary])

        # TP1/2/3가 이미 compute_quadrant_targets로 설정된 경우 그대로 사용 (1:2 RR 보존)
        if signal.tp1_price and signal.tp1_price > 0 and signal.tp2_price and signal.tp2_price > 0:
            return reordered_plan, signal.tp1_price, signal.tp2_price, signal.tp3_price

        # stop_price 기반 1:2 RR 직접 계산 (fallback)
        if signal.stop_price and signal.stop_price > 0:
            risk = abs(signal.entry_price - signal.stop_price)
            if signal.side == Side.LONG:
                tp1 = signal.entry_price + risk * 0.5
                tp2 = signal.entry_price + risk * 1.0
                tp3 = signal.entry_price + risk * 1.5
            else:
                tp1 = signal.entry_price - risk * 0.5
                tp2 = signal.entry_price - risk * 1.0
                tp3 = signal.entry_price - risk * 1.5
            return reordered_plan, tp1, tp2, tp3

        # 구조적 타겟 기반 fallback (stop_price 없을 때)
        if primary is None:
            return raw_plan, signal.tp1_price, signal.tp2_price, signal.tp3_price
        primary_price = float(primary["price"])
        delta = primary_price - signal.entry_price
        if signal.side == Side.LONG and delta <= 0:
            return raw_plan, signal.tp1_price, signal.tp2_price, signal.tp3_price
        if signal.side == Side.SHORT and delta >= 0:
            return raw_plan, signal.tp1_price, signal.tp2_price, signal.tp3_price
        tp1 = signal.entry_price + delta * 0.5
        tp2 = signal.entry_price + delta * 1.0
        tp3 = signal.entry_price + delta * 1.5
        return reordered_plan, tp1, tp2, tp3

    def _pick_primary_target(
        self,
        signal: StrategySignal,
        plan: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Pick the nearest valid structural anchor target for staged exits."""

        if not plan:
            return None
        return plan[0]
