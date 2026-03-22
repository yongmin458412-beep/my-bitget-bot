"""Expected-value gating."""

from __future__ import annotations

from dataclasses import dataclass

from core.settings import EVConfig

from .base import StrategySignal


@dataclass(slots=True)
class EVResult:
    """Expected value computation result."""

    expected_value: float
    p_win: float
    avg_win_r: float
    avg_loss_r: float
    fees_r: float
    slippage_r: float
    event_penalty: float
    funding_penalty: float
    spread_penalty: float
    approved: bool
    reject_reason: str = ""
    rr_to_tp1: float = 0.0
    rr_to_tp2: float = 0.0
    rr_to_best_target: float = 0.0


class ExpectedValueFilter:
    """Calculate whether a setup has sufficient edge after costs."""

    def __init__(self, config: EVConfig) -> None:
        self.config = config

    def evaluate(
        self,
        signal: StrategySignal,
        *,
        historical_win_rate: float,
        historical_avg_win_r: float,
        historical_avg_loss_r: float,
        spread_bps: float,
        funding_minutes_away: int | None,
        news_penalty: float,
    ) -> EVResult:
        """Return EV and pass/fail decision."""

        rr_to_tp1 = float(signal.rr_to_tp1 or signal.ev_metrics.get("rr_to_tp1", 0.0))
        rr_to_tp2 = float(signal.rr_to_tp2 or signal.ev_metrics.get("rr_to_tp2", 0.0))
        rr_to_best_target = float(signal.rr_to_best_target or signal.ev_metrics.get("rr_to_best_target", signal.expected_r))
        if signal.trade_rejected_reason:
            return EVResult(
                expected_value=-1.0,
                p_win=0.0,
                avg_win_r=rr_to_best_target,
                avg_loss_r=1.0,
                fees_r=signal.fees_r,
                slippage_r=signal.slippage_r,
                event_penalty=news_penalty,
                funding_penalty=0.0,
                spread_penalty=max(0.0, spread_bps / 10_000),
                approved=False,
                reject_reason=signal.trade_rejected_reason,
                rr_to_tp1=rr_to_tp1,
                rr_to_tp2=rr_to_tp2,
                rr_to_best_target=rr_to_best_target,
            )

        p_win = max(historical_win_rate, self.config.min_historical_win_rate)
        avg_win_r = max(historical_avg_win_r, rr_to_best_target or signal.expected_r)
        avg_loss_r = max(1.0, historical_avg_loss_r)
        fees_r = signal.fees_r
        slippage_r = signal.slippage_r
        event_penalty = news_penalty
        funding_penalty = 0.0
        if funding_minutes_away is not None and funding_minutes_away <= self.config.funding_penalty_minutes:
            funding_penalty = 0.08
        spread_penalty = max(0.0, spread_bps / 10_000)
        expected_value = (
            p_win * avg_win_r
            - (1 - p_win) * avg_loss_r
            - fees_r
            - slippage_r
            - event_penalty
            - funding_penalty
            - spread_penalty
        )
        approved = bool(signal.ev_metrics.get("approved", True)) and expected_value >= self.config.min_ev
        return EVResult(
            expected_value=expected_value,
            p_win=p_win,
            avg_win_r=avg_win_r,
            avg_loss_r=avg_loss_r,
            fees_r=fees_r,
            slippage_r=slippage_r,
            event_penalty=event_penalty,
            funding_penalty=funding_penalty,
            spread_penalty=spread_penalty,
            approved=approved,
            reject_reason="" if approved else (signal.trade_rejected_reason or "ev_filter"),
            rr_to_tp1=rr_to_tp1,
            rr_to_tp2=rr_to_tp2,
            rr_to_best_target=rr_to_best_target,
        )
