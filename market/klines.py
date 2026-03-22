"""Kline fetching and caching helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from exchange.bitget_demo import BitgetExchangeBase
from exchange.bitget_models import Candle


@dataclass(slots=True)
class CachedFrame:
    """Small in-memory cache entry."""

    dataframe: pd.DataFrame
    expires_at: datetime


class KlineService:
    """Fetch and cache klines from the exchange."""

    def __init__(self, exchange: BitgetExchangeBase) -> None:
        self.exchange = exchange
        self._cache: dict[tuple[str, str, str, int], CachedFrame] = {}

    async def get_dataframe(
        self,
        product_type: Any,
        symbol: str,
        timeframe: str,
        *,
        limit: int = 300,
        ttl_seconds: int = 10,
    ) -> pd.DataFrame:
        """Return a normalized pandas DataFrame indexed by timestamp."""

        key = (product_type.value, symbol.upper(), timeframe, limit)
        now = datetime.now(tz=UTC)
        cached = self._cache.get(key)
        if cached and cached.expires_at > now:
            return cached.dataframe.copy()

        candles = await self.exchange.rest.get_candles(product_type, symbol, timeframe, limit=limit)
        df = self._to_dataframe(candles)
        self._cache[key] = CachedFrame(dataframe=df, expires_at=now + timedelta(seconds=ttl_seconds))
        return df.copy()

    async def get_multi_timeframe(
        self,
        product_type: Any,
        symbol: str,
        timeframes: list[str],
        *,
        limit: int = 300,
    ) -> dict[str, pd.DataFrame]:
        """Return multiple timeframes for a symbol."""

        frames: dict[str, pd.DataFrame] = {}
        for timeframe in timeframes:
            frames[timeframe] = await self.get_dataframe(product_type, symbol, timeframe, limit=limit)
        return frames

    @staticmethod
    def _to_dataframe(candles: list[Candle]) -> pd.DataFrame:
        """Convert candle models to a typed DataFrame."""

        if not candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "quote_volume"])

        df = pd.DataFrame(
            {
                "timestamp": [item.timestamp for item in candles],
                "open": [item.open for item in candles],
                "high": [item.high for item in candles],
                "low": [item.low for item in candles],
                "close": [item.close for item in candles],
                "volume": [item.volume for item in candles],
                "quote_volume": [item.quote_volume for item in candles],
            }
        )
        return df.set_index("timestamp").sort_index()

