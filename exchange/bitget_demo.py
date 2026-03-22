"""Demo trading adapter that mirrors the live exchange interface."""

from __future__ import annotations

from typing import Any

from core.enums import ProductType, TradingMode
from core.settings import AppSettings

from .bitget_models import ContractConfig, OrderRequest, OrderResult
from .bitget_rest import BitgetRestClient
from .bitget_ws import BitgetWebSocketClient


class BitgetExchangeBase:
    """Common exchange adapter logic shared by demo and live modes."""

    def __init__(self, settings: AppSettings, *, demo: bool) -> None:
        self.settings = settings
        self.demo = demo
        secrets = settings.secrets
        if demo:
            api_key = secrets.bitget_demo_api_key or secrets.bitget_api_key
            api_secret = secrets.bitget_demo_api_secret or secrets.bitget_api_secret
            passphrase = secrets.bitget_demo_api_passphrase or secrets.bitget_api_passphrase
        else:
            api_key = secrets.bitget_api_key
            api_secret = secrets.bitget_api_secret
            passphrase = secrets.bitget_api_passphrase

        self.rest = BitgetRestClient(
            settings,
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
            demo=demo,
        )
        self.ws = BitgetWebSocketClient(
            settings,
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
            demo=demo,
        )

    @property
    def mode(self) -> TradingMode:
        """Return adapter mode."""

        return TradingMode.DEMO if self.demo else TradingMode.LIVE

    async def close(self) -> None:
        """Close REST and WS resources."""

        await self.ws.stop()
        await self.rest.aclose()

    async def get_all_contracts(self) -> list[ContractConfig]:
        """Fetch contract configs for every configured product type."""

        items: list[ContractConfig] = []
        for product_type in self.settings.exchange.product_types:
            items.extend(await self.rest.get_contracts(product_type))
        return items

    async def get_all_tickers(self) -> list[Any]:
        """Fetch tickers for every configured product type."""

        items: list[Any] = []
        for product_type in self.settings.exchange.product_types:
            items.extend(await self.rest.get_tickers(product_type))
        return items

    async def get_positions(self) -> list[Any]:
        """Fetch open positions across product types."""

        items: list[Any] = []
        for product_type in self.settings.exchange.product_types:
            items.extend(await self.rest.get_positions(product_type))
        return items

    async def get_accounts(self) -> list[Any]:
        """Fetch account balances across product types."""

        items: list[Any] = []
        for product_type in self.settings.exchange.product_types:
            items.extend(await self.rest.get_account_list(product_type))
        return items

    async def get_pending_orders(self) -> list[dict[str, Any]]:
        """Fetch pending orders across product types."""

        items: list[dict[str, Any]] = []
        for product_type in self.settings.exchange.product_types:
            rows = await self.rest.get_pending_orders(product_type)
            if rows:
                items.extend(rows)
        return items

    async def get_contract(self, symbol: str, product_type: ProductType) -> ContractConfig | None:
        """Fetch a single contract definition."""

        items = await self.rest.get_contracts(product_type, symbol=symbol)
        return items[0] if items else None

    async def place_order(self, order: OrderRequest) -> OrderResult:
        """Place an order through the REST client."""

        return await self.rest.place_order(order)


class BitgetDemoExchange(BitgetExchangeBase):
    """Demo trading exchange wrapper."""

    def __init__(self, settings: AppSettings) -> None:
        super().__init__(settings, demo=True)
