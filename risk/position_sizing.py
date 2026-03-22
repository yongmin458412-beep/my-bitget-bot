"""Position sizing based on stop distance and risk budget."""

from __future__ import annotations

from dataclasses import dataclass

from exchange.bitget_models import ContractConfig
from core.utils import round_to_step


@dataclass(slots=True)
class PositionSizeResult:
    """Position sizing output."""

    quantity: float
    risk_amount: float
    notional: float
    leverage: float
    valid: bool
    reason: str = ""


class PositionSizer:
    """Size positions from account equity and stop distance."""

    def size_from_risk(
        self,
        *,
        account_equity: float,
        risk_fraction: float,
        entry_price: float,
        stop_price: float,
        leverage: float,
        contract: ContractConfig,
    ) -> PositionSizeResult:
        """Calculate quantity using stop-distance-based risk."""

        risk_amount = account_equity * risk_fraction
        stop_distance = abs(entry_price - stop_price)
        if account_equity <= 0:
            return PositionSizeResult(0.0, 0.0, 0.0, leverage, False, "equity<=0")
        if stop_distance <= 0:
            return PositionSizeResult(0.0, risk_amount, 0.0, leverage, False, "stop_distance<=0")

        raw_quantity = risk_amount / stop_distance
        quantity = round_to_step(raw_quantity, contract.size_step or 1.0, mode="down")
        if contract.min_order_size and quantity < contract.min_order_size:
            quantity = round_to_step(contract.min_order_size, contract.size_step or 1.0, mode="up")
        notional = quantity * entry_price
        valid = quantity > 0
        reason = "" if valid else "quantity<=0"
        return PositionSizeResult(
            quantity=quantity,
            risk_amount=risk_amount,
            notional=notional,
            leverage=leverage,
            valid=valid,
            reason=reason,
        )

    def size_from_equity_share(
        self,
        *,
        account_equity: float,
        equity_share_count: int,
        entry_price: float,
        leverage: float,
        contract: ContractConfig,
    ) -> PositionSizeResult:
        """자산 균등 배분 방식: 총자산 / 포지션수 = 포지션당 증거금.

        예: 1000 USDT / 10포지션 = 100 USDT 증거금 × 20x 레버리지 = 2000 USDT 노션
        실제 손실은 손절 설정에 따라 달라지며, 증거금(100 USDT)이 최대 손실 한도.
        """
        if account_equity <= 0:
            return PositionSizeResult(0.0, 0.0, 0.0, leverage, False, "equity<=0")
        if entry_price <= 0:
            return PositionSizeResult(0.0, 0.0, 0.0, leverage, False, "entry_price<=0")
        if equity_share_count <= 0:
            return PositionSizeResult(0.0, 0.0, 0.0, leverage, False, "equity_share_count<=0")

        margin_per_position = account_equity / equity_share_count
        notional = margin_per_position * leverage
        raw_quantity = notional / entry_price

        quantity = round_to_step(raw_quantity, contract.size_step or 1.0, mode="down")
        if contract.min_order_size and quantity < contract.min_order_size:
            quantity = round_to_step(contract.min_order_size, contract.size_step or 1.0, mode="up")

        actual_notional = quantity * entry_price
        valid = quantity > 0
        reason = "" if valid else "quantity<=0"
        return PositionSizeResult(
            quantity=quantity,
            risk_amount=margin_per_position,   # 증거금 = 최대 손실 한도
            notional=actual_notional,
            leverage=leverage,
            valid=valid,
            reason=reason,
        )
