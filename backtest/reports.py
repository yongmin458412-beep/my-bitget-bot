"""Backtest report generation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from .simulator import SimulatedTrade


def build_report(trades: list[SimulatedTrade]) -> str:
    """Create a markdown report."""

    if not trades:
        return "# Backtest Report\n\n거래가 없습니다.\n"

    df = pd.DataFrame([trade.__dict__ for trade in trades])
    wins = df[df["pnl_usdt"] > 0]
    losses = df[df["pnl_usdt"] <= 0]
    total = df["pnl_usdt"].sum()
    win_rate = len(wins) / len(df)
    avg_win = wins["pnl_usdt"].mean() if not wins.empty else 0.0
    avg_loss = losses["pnl_usdt"].mean() if not losses.empty else 0.0
    profit_factor = wins["pnl_usdt"].sum() / abs(losses["pnl_usdt"].sum()) if not losses.empty and abs(losses["pnl_usdt"].sum()) > 0 else 0.0
    cumulative = df["pnl_usdt"].cumsum()
    max_drawdown = float((cumulative.cummax() - cumulative).max())
    expectancy = float(df["pnl_usdt"].mean())

    by_symbol = df.groupby("symbol")["pnl_usdt"].sum().sort_values(ascending=False)
    by_hour = df.assign(hour=pd.to_datetime(df["entry_time"]).dt.hour).groupby("hour")["pnl_usdt"].sum()
    by_strategy = df.groupby("strategy")["pnl_usdt"].agg(["sum", "count", "mean"])

    lines = [
        "# Backtest Report",
        "",
        f"- 총손익: {total:,.2f} USDT",
        f"- 승률: {win_rate * 100:.2f}%",
        f"- 평균 이익: {avg_win:,.2f}",
        f"- 평균 손실: {avg_loss:,.2f}",
        f"- Profit Factor: {profit_factor:.2f}",
        f"- Max Drawdown: {max_drawdown:,.2f}",
        f"- Expectancy: {expectancy:,.2f}",
        "",
        "## 심볼별 성과",
    ]
    lines.extend(f"- {symbol}: {value:,.2f}" for symbol, value in by_symbol.items())
    lines.append("")
    lines.append("## 시간대별 성과")
    lines.extend(f"- {hour:02d}시: {value:,.2f}" for hour, value in by_hour.items())
    lines.append("")
    lines.append("## 전략별 성과")
    lines.extend(
        f"- {strategy}: 합계 {row['sum']:,.2f} / 거래수 {int(row['count'])} / 평균 {row['mean']:,.2f}"
        for strategy, row in by_strategy.iterrows()
    )
    return "\n".join(lines) + "\n"


def write_report(output_path: str | Path, content: str) -> Path:
    """Write report to disk."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path

