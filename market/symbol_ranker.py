"""Active universe symbol ranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from exchange.bitget_models import ContractConfig, TickerSnapshot


@dataclass(slots=True)
class RankedSymbol:
    """Ranked symbol output."""

    symbol: str
    product_type: str
    total_score: float
    liquidity_score: float
    volatility_score: float
    spread_score: float
    depth_score: float
    trend_score: float
    anomaly_score: float
    age_score: float
    tradability_score: float
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_row(self, created_at: str) -> dict[str, Any]:
        """Return a persistence-friendly row."""

        return {
            "symbol": self.symbol,
            "product_type": self.product_type,
            "total_score": self.total_score,
            "liquidity_score": self.liquidity_score,
            "volatility_score": self.volatility_score,
            "spread_score": self.spread_score,
            "depth_score": self.depth_score,
            "trend_score": self.trend_score,
            "anomaly_score": self.anomaly_score,
            "age_score": self.age_score,
            "tradability_score": self.tradability_score,
            "metrics": self.metrics,
            "created_at": created_at,
        }


def _clip_01(value: float) -> float:
    """Clamp to the [0, 1] interval."""

    return max(0.0, min(1.0, value))


def rank_symbols(
    contracts: list[ContractConfig],
    tickers: list[TickerSnapshot],
    *,
    depth_metrics: dict[str, Any] | None = None,
    trend_metrics: dict[str, float] | None = None,
    active_universe_size: int = 50,
    min_quote_volume: float = 2_000_000,
    max_spread_bps: float = 8.0,
) -> list[RankedSymbol]:
    """Combine liquidity, volatility, spread, depth, trend, anomaly, and age scores."""

    depth_metrics = depth_metrics or {}
    trend_metrics = trend_metrics or {}
    ticker_by_symbol = {item.symbol: item for item in tickers}
    now = datetime.now(tz=UTC)
    ranked: list[RankedSymbol] = []

    max_turnover = max((item.turnover_24h for item in tickers), default=1.0)
    max_depth = max((getattr(item, "top5_notional", 0.0) for item in depth_metrics.values()), default=1.0)

    for contract in contracts:
        ticker = ticker_by_symbol.get(contract.symbol)
        if ticker is None:
            continue
        if ticker.turnover_24h < min_quote_volume:
            continue
        spread_score = _clip_01(1 - (ticker.spread_bps / max_spread_bps)) if max_spread_bps > 0 else 0.0
        if spread_score <= 0:
            continue

        liquidity_score = _clip_01(ticker.turnover_24h / max_turnover)
        volatility_score = _clip_01(abs(ticker.change_24h) / 0.08)
        depth_info = depth_metrics.get(contract.symbol)
        depth_value = getattr(depth_info, "top5_notional", 0.0)
        depth_score = _clip_01(depth_value / max_depth) if max_depth > 0 else 0.0
        trend_score = _clip_01(abs(trend_metrics.get(contract.symbol, 0.0)))
        anomaly_score = _clip_01(1 - (abs(ticker.change_24h) / 0.15))
        if contract.listing_time:
            age_hours = max(0.0, (now - contract.listing_time).total_seconds() / 3600)
        else:
            age_hours = 9999.0
        age_score = _clip_01(age_hours / 168)
        tradability_score = 1.0 if contract.status.lower() in {"normal", "listed"} else 0.0

        total = (
            liquidity_score * 0.28
            + volatility_score * 0.18
            + spread_score * 0.18
            + depth_score * 0.14
            + trend_score * 0.12
            + anomaly_score * 0.05
            + age_score * 0.03
            + tradability_score * 0.02
        )
        ranked.append(
            RankedSymbol(
                symbol=contract.symbol,
                product_type=contract.product_type.value,
                total_score=round(total, 6),
                liquidity_score=round(liquidity_score, 6),
                volatility_score=round(volatility_score, 6),
                spread_score=round(spread_score, 6),
                depth_score=round(depth_score, 6),
                trend_score=round(trend_score, 6),
                anomaly_score=round(anomaly_score, 6),
                age_score=round(age_score, 6),
                tradability_score=round(tradability_score, 6),
                metrics={
                    "spread_bps": ticker.spread_bps,
                    "turnover_24h": ticker.turnover_24h,
                    "depth_top5_notional": depth_value,
                    "change_24h": ticker.change_24h,
                },
            )
        )

    return sorted(ranked, key=lambda item: item.total_score, reverse=True)[: max(1, active_universe_size)]

