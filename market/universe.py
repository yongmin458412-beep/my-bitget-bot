"""Universe management for full-symbol monitoring and active tradable selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from core.persistence import SQLitePersistence
from core.settings import AppSettings
from core.state_store import StateStore
from core.utils import unique_preserve_order
from exchange.bitget_demo import BitgetExchangeBase
from exchange.bitget_models import ContractConfig

from .klines import KlineService
from .orderbook import OrderBookService
from .symbol_ranker import RankedSymbol, rank_symbols


@dataclass(slots=True)
class UniverseSnapshot:
    """Universe output used by runtime and dashboard layers."""

    tracked_contracts: list[ContractConfig] = field(default_factory=list)
    active_symbols: list[str] = field(default_factory=list)
    ranked_symbols: list[RankedSymbol] = field(default_factory=list)
    refreshed_at: str = ""


class UniverseManager:
    """Refresh all futures symbols and maintain an active tradable universe."""

    def __init__(
        self,
        settings: AppSettings,
        exchange: BitgetExchangeBase,
        persistence: SQLitePersistence,
        state_store: StateStore,
        klines: KlineService,
        orderbooks: OrderBookService,
    ) -> None:
        self.settings = settings
        self.exchange = exchange
        self.persistence = persistence
        self.state_store = state_store
        self.klines = klines
        self.orderbooks = orderbooks

    async def refresh(self) -> UniverseSnapshot:
        """Refresh all contracts and re-rank the active universe."""

        contracts = await self.exchange.get_all_contracts()
        tickers = await self.exchange.get_all_tickers()
        trend_metrics: dict[str, float] = {}
        depth_metrics: dict[str, Any] = {}

        candidate_symbols = [
            ticker.symbol
            for ticker in sorted(tickers, key=lambda item: item.turnover_24h, reverse=True)
        ]
        for symbol in candidate_symbols[: min(len(candidate_symbols), self.settings.universe.active_universe_size * 2)]:
            contract = next((item for item in contracts if item.symbol == symbol), None)
            if contract is None:
                continue
            try:
                frame = await self.klines.get_dataframe(contract.product_type, symbol, self.settings.timeframes.confirm, limit=120)
                if not frame.empty:
                    close = frame["close"]
                    trend_metrics[symbol] = float((close.iloc[-1] / close.iloc[0]) - 1)
                depth_metrics[symbol] = await self.orderbooks.get_metrics(contract.product_type, symbol)
            except Exception:  # noqa: BLE001
                continue

        ranked = rank_symbols(
            contracts,
            tickers,
            depth_metrics=depth_metrics,
            trend_metrics=trend_metrics,
            active_universe_size=self.settings.universe.active_universe_size,
            min_quote_volume=self.settings.universe.min_24h_quote_volume,
            max_spread_bps=self.settings.universe.max_spread_bps,
        )
        active_symbols = [item.symbol for item in ranked]
        if self.settings.universe.include_btc_eth_always:
            active_symbols = unique_preserve_order(["BTCUSDT", "ETHUSDT", *active_symbols])
            active_symbols = active_symbols[: self.settings.universe.active_universe_size]

        refreshed_at = datetime.now(tz=UTC).isoformat(timespec="seconds")
        self.persistence.save_symbol_scores([item.to_row(refreshed_at) for item in ranked])
        self.state_store.set_tracked_symbols([item.symbol for item in contracts])
        self.state_store.set_active_universe(active_symbols)
        return UniverseSnapshot(
            tracked_contracts=contracts,
            active_symbols=active_symbols,
            ranked_symbols=ranked,
            refreshed_at=refreshed_at,
        )
