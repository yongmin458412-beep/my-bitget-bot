"""Order book helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from exchange.bitget_demo import BitgetExchangeBase
from exchange.bitget_models import OrderBookSnapshot


@dataclass(slots=True)
class DepthMetrics:
    """Summary metrics derived from an order book snapshot."""

    symbol: str
    spread_bps: float
    top5_notional: float
    imbalance: float

    def to_payload(self) -> dict[str, float | str]:
        """Return a JSON-serializable payload."""

        return asdict(self)


class OrderBookService:
    """Fetch and summarize order book depth."""

    def __init__(self, exchange: BitgetExchangeBase) -> None:
        self.exchange = exchange

    async def get_snapshot(self, product_type: Any, symbol: str) -> OrderBookSnapshot:
        """Fetch a merged depth snapshot."""

        return await self.exchange.rest.get_merge_depth(product_type, symbol)

    async def get_metrics(self, product_type: Any, symbol: str) -> DepthMetrics:
        """Fetch a snapshot and derive simple liquidity metrics."""

        snapshot = await self.get_snapshot(product_type, symbol)
        bid_depth = sum(level.price * level.size for level in snapshot.bids[:5])
        ask_depth = sum(level.price * level.size for level in snapshot.asks[:5])
        imbalance = 0.0
        if bid_depth + ask_depth > 0:
            imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
        return DepthMetrics(
            symbol=symbol.upper(),
            spread_bps=snapshot.spread_bps,
            top5_notional=snapshot.depth_notional_top5,
            imbalance=imbalance,
        )
