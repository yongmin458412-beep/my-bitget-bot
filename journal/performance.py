"""Performance summaries for Telegram and Streamlit."""

from __future__ import annotations

from dataclasses import dataclass

from core.persistence import SQLitePersistence


@dataclass(slots=True)
class PerformanceSnapshot:
    """Aggregate performance output."""

    realized_pnl: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    max_drawdown: float
    trade_count: int


class PerformanceAnalyzer:
    """Compute performance metrics from journal rows."""

    def __init__(self, persistence: SQLitePersistence) -> None:
        self.persistence = persistence

    def summarize(self) -> PerformanceSnapshot:
        """Return overall performance metrics."""

        rows = self.persistence.fetchall("SELECT * FROM trades WHERE status = 'closed'")
        if not rows:
            return PerformanceSnapshot(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        pnls = [float(row.get("realized_pnl_usdt") or 0.0) for row in rows]
        wins = [value for value in pnls if value > 0]
        losses = [value for value in pnls if value < 0]
        realized_pnl = sum(pnls)
        trade_count = len(pnls)
        win_rate = len(wins) / trade_count if trade_count else 0.0
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
        profit_factor = (sum(wins) / abs(sum(losses))) if losses else float("inf")
        expectancy = realized_pnl / trade_count if trade_count else 0.0
        peak = 0.0
        cumulative = 0.0
        max_drawdown = 0.0
        for pnl in pnls:
            cumulative += pnl
            peak = max(peak, cumulative)
            max_drawdown = min(max_drawdown, cumulative - peak)
        return PerformanceSnapshot(
            realized_pnl=realized_pnl,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor if profit_factor != float("inf") else 0.0,
            expectancy=expectancy,
            max_drawdown=abs(max_drawdown),
            trade_count=trade_count,
        )
