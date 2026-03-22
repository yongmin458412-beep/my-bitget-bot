"""Typed Bitget exchange models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from core.enums import OrderStatus, OrderType, ProductType, Side, TimeInForce
from core.utils import as_float


def _ts_to_datetime(value: str | int | float | None) -> datetime | None:
    """Convert exchange millisecond timestamps to datetime."""

    if value in (None, "", 0, "0"):
        return None
    try:
        millis = int(float(value))
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(millis / 1000, tz=UTC)


class ContractConfig(BaseModel):
    """Normalized contract metadata."""

    symbol: str
    product_type: ProductType
    base_coin: str
    quote_coin: str
    margin_coin: str
    min_order_size: float = 0.0
    size_step: float = 0.0
    price_step: float = 0.0
    price_precision: int = 0
    size_precision: int = 0
    min_leverage: float | None = None
    max_leverage: float | None = None
    maker_fee_rate: float | None = None
    taker_fee_rate: float | None = None
    status: str = "normal"
    listing_time: datetime | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_api(cls, product_type: ProductType, payload: dict[str, Any]) -> "ContractConfig":
        """Build a contract config from Bitget REST payload."""

        return cls(
            symbol=str(payload.get("symbol", "")).upper(),
            product_type=product_type,
            base_coin=str(payload.get("baseCoin", "")),
            quote_coin=str(payload.get("quoteCoin", "")),
            margin_coin=str(payload.get("marginCoin", payload.get("quoteCoin", ""))),
            min_order_size=as_float(
                payload.get("minTradeNum") or payload.get("minTradeUSDT") or payload.get("minTradeAmount")
            ),
            size_step=as_float(payload.get("sizeMultiplier") or payload.get("minTradeNum")),
            price_step=as_float(payload.get("priceMultiplier") or payload.get("pricePlace")),
            price_precision=int(as_float(payload.get("pricePlace"))),
            size_precision=int(as_float(payload.get("volumePlace"))),
            min_leverage=as_float(payload.get("minLever")) or None,
            max_leverage=as_float(payload.get("maxLever")) or None,
            maker_fee_rate=as_float(payload.get("makerFeeRate")) or None,
            taker_fee_rate=as_float(payload.get("takerFeeRate")) or None,
            status=str(payload.get("symbolStatus", payload.get("status", "normal"))),
            listing_time=_ts_to_datetime(payload.get("launchTime") or payload.get("offTime")),
            raw=payload,
        )


class TickerSnapshot(BaseModel):
    """Normalized ticker snapshot."""

    symbol: str
    product_type: ProductType
    last_price: float
    bid_price: float
    ask_price: float
    bid_size: float = 0.0
    ask_size: float = 0.0
    mark_price: float = 0.0
    index_price: float = 0.0
    funding_rate: float = 0.0
    open_interest: float = 0.0
    volume_24h: float = 0.0
    turnover_24h: float = 0.0
    change_24h: float = 0.0
    timestamp: datetime | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def spread_bps(self) -> float:
        """Bid/ask spread in basis points."""

        if self.ask_price <= 0 or self.bid_price <= 0:
            return 0.0
        mid = (self.ask_price + self.bid_price) / 2
        if mid <= 0:
            return 0.0
        return ((self.ask_price - self.bid_price) / mid) * 10_000

    @classmethod
    def from_api(cls, product_type: ProductType, payload: dict[str, Any]) -> "TickerSnapshot":
        """Build a ticker snapshot from Bitget REST or WS payload."""

        return cls(
            symbol=str(payload.get("symbol") or payload.get("instId", "")).upper(),
            product_type=product_type,
            last_price=as_float(payload.get("lastPr") or payload.get("lastPrice")),
            bid_price=as_float(payload.get("bidPr") or payload.get("bid1Price")),
            ask_price=as_float(payload.get("askPr") or payload.get("ask1Price")),
            bid_size=as_float(payload.get("bidSz") or payload.get("bid1Size")),
            ask_size=as_float(payload.get("askSz") or payload.get("ask1Size")),
            mark_price=as_float(payload.get("markPrice")),
            index_price=as_float(payload.get("indexPrice")),
            funding_rate=as_float(payload.get("fundingRate")),
            open_interest=as_float(payload.get("openInterest")),
            volume_24h=as_float(payload.get("baseVolume") or payload.get("volume24h")),
            turnover_24h=as_float(payload.get("quoteVolume") or payload.get("turnover24h") or payload.get("usdtVolume")),
            change_24h=as_float(payload.get("change24h") or payload.get("price24hPcnt")),
            timestamp=_ts_to_datetime(payload.get("ts")),
            raw=payload,
        )


class Candle(BaseModel):
    """Normalized candle."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float

    @classmethod
    def from_api(cls, payload: list[str]) -> "Candle":
        """Build a candle from Bitget REST array format."""

        return cls(
            timestamp=_ts_to_datetime(payload[0]) or datetime.now(tz=UTC),
            open=as_float(payload[1]),
            high=as_float(payload[2]),
            low=as_float(payload[3]),
            close=as_float(payload[4]),
            volume=as_float(payload[5]),
            quote_volume=as_float(payload[6]),
        )


class OrderBookLevel(BaseModel):
    """Single order book price level."""

    price: float
    size: float


class OrderBookSnapshot(BaseModel):
    """Normalized order book snapshot."""

    symbol: str
    product_type: ProductType
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    scale: float = 0.0
    timestamp: datetime | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def spread_bps(self) -> float:
        """Return top-of-book spread."""

        if not self.bids or not self.asks:
            return 0.0
        bid = self.bids[0].price
        ask = self.asks[0].price
        if bid <= 0 or ask <= 0:
            return 0.0
        mid = (bid + ask) / 2
        return ((ask - bid) / mid) * 10_000

    @property
    def depth_notional_top5(self) -> float:
        """Approximate depth in quote currency for the first five levels."""

        top_bids = sum(level.price * level.size for level in self.bids[:5])
        top_asks = sum(level.price * level.size for level in self.asks[:5])
        return top_bids + top_asks

    @classmethod
    def from_api(
        cls,
        symbol: str,
        product_type: ProductType,
        payload: dict[str, Any],
    ) -> "OrderBookSnapshot":
        """Build from REST or WS order book payload."""

        return cls(
            symbol=symbol.upper(),
            product_type=product_type,
            bids=[OrderBookLevel(price=as_float(row[0]), size=as_float(row[1])) for row in payload.get("bids", [])],
            asks=[OrderBookLevel(price=as_float(row[0]), size=as_float(row[1])) for row in payload.get("asks", [])],
            scale=as_float(payload.get("scale")),
            timestamp=_ts_to_datetime(payload.get("ts")),
            raw=payload,
        )


class AccountSummary(BaseModel):
    """Normalized account information."""

    product_type: ProductType
    margin_coin: str
    available: float
    locked: float
    account_equity: float
    usdt_equity: float
    unrealized_pnl: float
    crossed_margin: float
    isolated_margin: float
    asset_mode: str = "single"
    raw: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_api(cls, product_type: ProductType, payload: dict[str, Any]) -> "AccountSummary":
        """Build from account API response."""

        return cls(
            product_type=product_type,
            margin_coin=str(payload.get("marginCoin", "")),
            available=as_float(payload.get("available") or payload.get("unionAvailable")),
            locked=as_float(payload.get("locked")),
            account_equity=as_float(payload.get("accountEquity")),
            usdt_equity=as_float(payload.get("usdtEquity") or payload.get("accountEquity")),
            unrealized_pnl=as_float(payload.get("unrealizedPL") or payload.get("crossedUnrealizedPL")),
            crossed_margin=as_float(payload.get("crossedMargin")),
            isolated_margin=as_float(payload.get("isolatedMargin")),
            asset_mode=str(payload.get("assetMode", "single")),
            raw=payload,
        )


class PositionSnapshot(BaseModel):
    """Normalized futures position."""

    symbol: str
    product_type: ProductType
    side: Side
    size: float
    entry_price: float
    mark_price: float
    break_even_price: float = 0.0
    take_profit: float = 0.0
    stop_loss: float = 0.0
    leverage: float = 0.0
    liquidation_price: float = 0.0
    used_margin: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    margin_coin: str = ""
    opened_at: datetime | None = None
    updated_at: datetime | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_api(cls, product_type: ProductType, payload: dict[str, Any]) -> "PositionSnapshot":
        """Build from position payload."""

        hold_side = str(payload.get("holdSide", payload.get("posSide", payload.get("side", "long")))).lower()
        side = Side.LONG if hold_side in {"long", "buy"} else Side.SHORT
        return cls(
            symbol=str(payload.get("symbol", "")).upper(),
            product_type=product_type,
            side=side,
            size=as_float(payload.get("total") or payload.get("size") or payload.get("available")),
            entry_price=as_float(payload.get("openPriceAvg") or payload.get("avgOpenPrice")),
            mark_price=as_float(payload.get("markPrice")),
            break_even_price=as_float(payload.get("breakEvenPrice")),
            take_profit=as_float(payload.get("takeProfit")),
            stop_loss=as_float(payload.get("stopLoss")),
            leverage=as_float(payload.get("leverage")),
            liquidation_price=as_float(payload.get("liquidationPrice")),
            used_margin=as_float(payload.get("marginSize") or payload.get("margin")),
            unrealized_pnl=as_float(payload.get("unrealizedPL")),
            realized_pnl=as_float(payload.get("achievedProfits")),
            margin_coin=str(payload.get("marginCoin", "")),
            opened_at=_ts_to_datetime(payload.get("cTime")),
            updated_at=_ts_to_datetime(payload.get("uTime")),
            raw=payload,
        )


class OrderRequest(BaseModel):
    """Input payload for placing orders."""

    symbol: str
    product_type: ProductType
    margin_coin: str
    side: str
    trade_side: str | None = "open"
    order_type: OrderType = OrderType.LIMIT
    quantity: float
    price: float | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    client_order_id: str | None = None
    reduce_only: bool = False
    preset_take_profit_price: float | None = None
    preset_stop_loss_price: float | None = None
    preset_take_profit_execute_price: float | None = None
    preset_stop_loss_execute_price: float | None = None
    margin_mode: str = "isolated"
    stp_mode: str = "none"
    dry_run_reason: str | None = None

    def as_api_payload(self) -> dict[str, str]:
        """Serialize into Bitget's order payload."""

        payload: dict[str, str] = {
            "symbol": self.symbol.upper(),
            "productType": self.product_type.value,
            "marginMode": self.margin_mode,
            "marginCoin": self.margin_coin.upper(),
            "size": f"{self.quantity}",
            "side": self.side,
            "orderType": self.order_type.value,
            "stpMode": self.stp_mode,
        }
        if self.trade_side is not None:
            payload["tradeSide"] = self.trade_side
        if self.price is not None:
            payload["price"] = f"{self.price}"
        if self.order_type == OrderType.LIMIT:
            payload["force"] = self.time_in_force.value
        if self.client_order_id:
            payload["clientOid"] = self.client_order_id
        payload["reduceOnly"] = "YES" if self.reduce_only else "NO"
        if self.preset_take_profit_price is not None:
            payload["presetStopSurplusPrice"] = f"{self.preset_take_profit_price}"
        if self.preset_stop_loss_price is not None:
            payload["presetStopLossPrice"] = f"{self.preset_stop_loss_price}"
        if self.preset_take_profit_execute_price is not None:
            payload["presetStopSurplusExecutePrice"] = f"{self.preset_take_profit_execute_price}"
        if self.preset_stop_loss_execute_price is not None:
            payload["presetStopLossExecutePrice"] = f"{self.preset_stop_loss_execute_price}"
        return payload


class OrderResult(BaseModel):
    """Normalized order response."""

    client_order_id: str
    exchange_order_id: str | None
    status: OrderStatus = OrderStatus.NEW
    symbol: str
    product_type: ProductType
    raw: dict[str, Any] = Field(default_factory=dict)


class FillDetail(BaseModel):
    """Normalized fill detail record."""

    trade_id: str
    order_id: str
    symbol: str
    price: float
    size: float
    side: str
    fee: float
    pnl: float
    timestamp: datetime | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_api(cls, payload: dict[str, Any]) -> "FillDetail":
        """Build from fills API row."""

        return cls(
            trade_id=str(payload.get("tradeId", "")),
            order_id=str(payload.get("orderId", "")),
            symbol=str(payload.get("symbol", "")).upper(),
            price=as_float(payload.get("priceAvg") or payload.get("fillPrice") or payload.get("price")),
            size=as_float(payload.get("sizeQty") or payload.get("size")),
            side=str(payload.get("side", "")),
            fee=as_float(payload.get("fee")),
            pnl=as_float(payload.get("profit")),
            timestamp=_ts_to_datetime(payload.get("cTime") or payload.get("fillTime")),
            raw=payload,
        )

