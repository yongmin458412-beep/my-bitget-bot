"""Order routing logic."""

from __future__ import annotations

from dataclasses import dataclass

from core.enums import OrderType, Side, TimeInForce
from core.settings import AppSettings
from core.utils import round_to_step
from exchange.bitget_models import ContractConfig, OrderRequest
from risk.risk_engine import ApprovedTrade


@dataclass(slots=True)
class RoutingDecision:
    """Final routing plan for an order."""

    order: OrderRequest
    maker_attempt: bool
    fallback_allowed: bool
    rationale: list[str]


class OrderRouter:
    """Build orders with a maker-first policy."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def build_entry_order(
        self,
        *,
        approved_trade: ApprovedTrade,
        contract: ContractConfig,
        best_bid: float,
        best_ask: float,
        spread_bps: float,
        volatility_score: float,
    ) -> RoutingDecision:
        """Route an approved trade into a post-only or market order."""

        signal = approved_trade.signal
        rationale: list[str] = []
        maker_allowed = self.settings.execution.maker_first and spread_bps <= self.settings.universe.max_spread_bps
        use_taker = False
        if volatility_score >= self.settings.execution.taker_fallback_volatility_threshold and self.settings.execution.taker_fallback:
            use_taker = True
            rationale.append("급변장 taker fallback 허용")

        if maker_allowed and not use_taker:
            if signal.side == Side.LONG:
                price = round_to_step(best_bid, contract.price_step, mode="down") if contract.price_step else best_bid
                order_side = "buy"
            else:
                price = round_to_step(best_ask, contract.price_step, mode="up") if contract.price_step else best_ask
                order_side = "sell"
            # price_step가 너무 커서 반올림 결과가 0이 되면 원래 가격 사용
            if not price or price <= 0:
                price = best_bid if signal.side == Side.LONG else best_ask
            order = OrderRequest(
                symbol=signal.symbol,
                product_type=contract.product_type,
                margin_coin=contract.margin_coin,
                side=order_side,
                trade_side="open",
                order_type=OrderType.LIMIT,
                quantity=approved_trade.quantity,
                price=price,
                time_in_force=TimeInForce.POST_ONLY,
                preset_stop_loss_price=approved_trade.stop_price if self.settings.execution.use_exchange_plan_orders else None,
            )
            rationale.append("maker-first 지정가")
            return RoutingDecision(order=order, maker_attempt=True, fallback_allowed=self.settings.execution.taker_fallback, rationale=rationale)

        order = OrderRequest(
            symbol=signal.symbol,
            product_type=contract.product_type,
            margin_coin=contract.margin_coin,
            side="buy" if signal.side == Side.LONG else "sell",
            trade_side="open",
            order_type=OrderType.MARKET,
            quantity=approved_trade.quantity,
            preset_stop_loss_price=approved_trade.stop_price if self.settings.execution.use_exchange_plan_orders else None,
        )
        rationale.append("시장가 fallback")
        return RoutingDecision(order=order, maker_attempt=False, fallback_allowed=False, rationale=rationale)
