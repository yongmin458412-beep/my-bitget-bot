"""Position sizing based on stop distance and risk budget.

Sizing modes:
  - equity_share: 자산 균등 배분 (총자산 / N)
  - risk_based: 손절 거리 기반 (자본의 X% 리스크)
  - atr_proportional: ATR × 배수 = 손절 거리, 리스크 기반 사이징
  - kelly: Kelly Criterion 기반 (fractional Kelly)
"""

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
        # 레버리지 기반 최대 포지션 크기 제한: margin = notional / leverage <= equity * 0.95
        max_notional = account_equity * leverage * 0.95
        max_qty_by_margin = max_notional / entry_price if entry_price > 0 else raw_quantity
        raw_quantity = min(raw_quantity, max_qty_by_margin)
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

    def size_from_atr(
        self,
        *,
        account_equity: float,
        risk_fraction: float,
        entry_price: float,
        atr_value: float,
        atr_multiplier: float,
        leverage: float,
        contract: ContractConfig,
    ) -> PositionSizeResult:
        """ATR 비례 사이징: 손절 = ATR × multiplier."""
        stop_distance = atr_value * atr_multiplier
        if stop_distance <= 0:
            return PositionSizeResult(0.0, 0.0, 0.0, leverage, False, "atr_stop<=0")
        stop_price = entry_price - stop_distance  # 방향 무관, 거리만 사용
        return self.size_from_risk(
            account_equity=account_equity,
            risk_fraction=risk_fraction,
            entry_price=entry_price,
            stop_price=stop_price,
            leverage=leverage,
            contract=contract,
        )

    def size_from_kelly(
        self,
        *,
        account_equity: float,
        win_rate: float,
        avg_win_r: float,
        avg_loss_r: float,
        kelly_fraction: float,
        entry_price: float,
        stop_price: float,
        leverage: float,
        contract: ContractConfig,
    ) -> PositionSizeResult:
        """Kelly Criterion 기반 사이징 (fractional Kelly)."""
        if avg_loss_r <= 0 or win_rate <= 0:
            return self.size_from_risk(
                account_equity=account_equity,
                risk_fraction=0.005,
                entry_price=entry_price,
                stop_price=stop_price,
                leverage=leverage,
                contract=contract,
            )
        b = avg_win_r / avg_loss_r
        kelly_f = (win_rate * b - (1.0 - win_rate)) / b
        kelly_f = max(0.0, kelly_f * kelly_fraction)
        risk_fraction = min(kelly_f, 0.02)  # 최대 2% 캡
        return self.size_from_risk(
            account_equity=account_equity,
            risk_fraction=risk_fraction,
            entry_price=entry_price,
            stop_price=stop_price,
            leverage=leverage,
            contract=contract,
        )
