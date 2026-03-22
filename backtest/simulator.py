"""Trade simulation with partial TP and break-even logic."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from core.enums import Side
from risk.risk_engine import ApprovedTrade


@dataclass(slots=True)
class SimulatedTrade:
    """Completed backtest trade."""

    symbol: str
    strategy: str
    side: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl_usdt: float
    pnl_r: float
    fees_usdt: float
    slippage_usdt: float
    exit_reason: str
    tp1_hit: bool


@dataclass(slots=True)
class ActiveTrade:
    """Open simulated trade state."""

    approved: ApprovedTrade
    entry_time: datetime
    remaining_quantity: float
    stop_price: float
    initial_risk: float
    realized_pnl_usdt: float = 0.0
    realized_fees_usdt: float = 0.0
    realized_slippage_usdt: float = 0.0
    tp1_done: bool = False
    tp2_done: bool = False


class TradeSimulator:
    """Single-position simulator."""

    def __init__(
        self,
        fee_bps: float,
        slippage_bps: float,
        tp1_partial_close_pct: float,
        tp2_partial_close_pct: float,
        move_sl_to_be: bool,
        be_offset_r: float,
        max_hold_minutes: int,
    ) -> None:
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.tp1_partial_close_pct = tp1_partial_close_pct
        self.tp2_partial_close_pct = tp2_partial_close_pct
        self.move_sl_to_be = move_sl_to_be
        self.be_offset_r = be_offset_r
        self.max_hold_minutes = max_hold_minutes
        self.active_trade: ActiveTrade | None = None
        self.closed_trades: list[SimulatedTrade] = []

    def open_trade(self, approved: ApprovedTrade, entry_time: datetime) -> None:
        """Open a simulated trade."""

        self.active_trade = ActiveTrade(
            approved=approved,
            entry_time=entry_time,
            remaining_quantity=approved.quantity,
            stop_price=approved.stop_price,
            initial_risk=max(abs(approved.signal.entry_price - approved.stop_price), 1e-9),
        )

    def on_bar(self, candle: pd.Series) -> None:
        """Advance the simulator by one bar."""

        trade = self.active_trade
        if trade is None:
            return

        signal = trade.approved.signal
        high = float(candle["high"])
        low = float(candle["low"])
        close = float(candle["close"])
        if signal.side == Side.LONG:
            stop_hit = low <= trade.stop_price
            tp1_hit = high >= trade.approved.tp1_price
            tp2_hit = high >= trade.approved.tp2_price
            tp3_hit = trade.approved.tp3_price is not None and high >= trade.approved.tp3_price
        else:
            stop_hit = high >= trade.stop_price
            tp1_hit = low <= trade.approved.tp1_price
            tp2_hit = low <= trade.approved.tp2_price
            tp3_hit = trade.approved.tp3_price is not None and low <= trade.approved.tp3_price

        if tp1_hit and not trade.tp1_done:
            trade.tp1_done = True
            close_qty = min(trade.remaining_quantity, trade.approved.quantity * self.tp1_partial_close_pct)
            self._realize_partial(trade, trade.approved.tp1_price, close_qty)
            trade.remaining_quantity = max(0.0, trade.remaining_quantity - close_qty)
            if self.move_sl_to_be:
                if signal.side == Side.LONG:
                    trade.stop_price = signal.entry_price + trade.initial_risk * self.be_offset_r
                else:
                    trade.stop_price = signal.entry_price - trade.initial_risk * self.be_offset_r

        if tp2_hit and not trade.tp2_done and trade.remaining_quantity > 0:
            trade.tp2_done = True
            if trade.approved.tp3_price is None:
                self._close_trade(
                    exit_time=candle.name.to_pydatetime(),
                    exit_price=trade.approved.tp2_price,
                    reason="tp2",
                    tp1_hit=trade.tp1_done,
                )
                return
            close_qty = min(trade.remaining_quantity, trade.approved.quantity * self.tp2_partial_close_pct)
            self._realize_partial(trade, trade.approved.tp2_price, close_qty)
            trade.remaining_quantity = max(0.0, trade.remaining_quantity - close_qty)

        if tp3_hit and trade.remaining_quantity > 0:
            self._close_trade(
                exit_time=candle.name.to_pydatetime(),
                exit_price=trade.approved.tp3_price or trade.approved.tp2_price,
                reason="tp3",
                tp1_hit=trade.tp1_done,
            )
            return
        if stop_hit:
            self._close_trade(
                exit_time=candle.name.to_pydatetime(),
                exit_price=trade.stop_price,
                reason="stop",
                tp1_hit=trade.tp1_done,
            )
            return

        hold_minutes = (candle.name.to_pydatetime() - trade.entry_time).total_seconds() / 60
        if hold_minutes >= self.max_hold_minutes:
            self._close_trade(
                exit_time=candle.name.to_pydatetime(),
                exit_price=close,
                reason="time_stop",
                tp1_hit=trade.tp1_done,
            )

    def finalize(self, last_candle: pd.Series) -> None:
        """Force-close any remaining trade at the end of the test."""

        if self.active_trade is None:
            return
        self._close_trade(
            exit_time=last_candle.name.to_pydatetime(),
            exit_price=float(last_candle["close"]),
            reason="end_of_test",
            tp1_hit=self.active_trade.tp1_done,
        )

    def _close_trade(self, *, exit_time: datetime, exit_price: float, reason: str, tp1_hit: bool) -> None:
        """Close the active trade and persist the result."""

        trade = self.active_trade
        if trade is None:
            return
        signal = trade.approved.signal
        quantity = trade.remaining_quantity
        entry_price = signal.entry_price
        gross_pnl = (
            (exit_price - entry_price) * quantity
            if signal.side == Side.LONG
            else (entry_price - exit_price) * quantity
        )
        fees = quantity * entry_price * (self.fee_bps / 10_000) * 2
        slippage = quantity * entry_price * (self.slippage_bps / 10_000)
        pnl = gross_pnl - fees - slippage + trade.realized_pnl_usdt
        total_fees = fees + trade.realized_fees_usdt
        total_slippage = slippage + trade.realized_slippage_usdt
        pnl_r = pnl / max(trade.initial_risk * trade.approved.quantity, 1e-9)
        self.closed_trades.append(
            SimulatedTrade(
                symbol=signal.symbol,
                strategy=signal.strategy.value,
                side=signal.side.value,
                entry_time=trade.entry_time,
                exit_time=exit_time,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=trade.approved.quantity,
                pnl_usdt=pnl,
                pnl_r=pnl_r,
                fees_usdt=total_fees,
                slippage_usdt=total_slippage,
                exit_reason=reason,
                tp1_hit=tp1_hit,
            )
        )
        self.active_trade = None

    def _realize_partial(self, trade: ActiveTrade, exit_price: float, quantity: float) -> None:
        """Realize partial PnL before the final close."""

        if quantity <= 0:
            return
        signal = trade.approved.signal
        gross_pnl = (
            (exit_price - signal.entry_price) * quantity
            if signal.side == Side.LONG
            else (signal.entry_price - exit_price) * quantity
        )
        fees = quantity * signal.entry_price * (self.fee_bps / 10_000) * 2
        slippage = quantity * signal.entry_price * (self.slippage_bps / 10_000)
        trade.realized_pnl_usdt += gross_pnl - fees - slippage
        trade.realized_fees_usdt += fees
        trade.realized_slippage_usdt += slippage
