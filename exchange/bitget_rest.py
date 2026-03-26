"""Async Bitget REST client for futures market data and trading."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any
from urllib.parse import urlencode

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from core.enums import OrderStatus, ProductType
from core.logger import get_logger
from core.settings import AppSettings
from core.utils import as_float, round_to_step

from .bitget_models import (
    AccountSummary,
    Candle,
    ContractConfig,
    FillDetail,
    OrderBookSnapshot,
    OrderRequest,
    OrderResult,
    PositionSnapshot,
    TickerSnapshot,
)


class BitgetAPIError(RuntimeError):
    """Raised when Bitget returns a non-success response."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.payload = payload or {}


_TIMESTAMP_EXPIRED_CODES = {"40018", "40007", "40025"}


class BitgetRestClient:
    """Thin async wrapper around Bitget Futures REST APIs."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        api_key: str,
        api_secret: str,
        passphrase: str,
        demo: bool = True,
    ) -> None:
        self.settings = settings
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.demo = demo
        self.logger = get_logger(__name__)
        self._client = httpx.AsyncClient(
            base_url=settings.exchange.base_url,
            timeout=settings.exchange.request_timeout_seconds,
            headers={"Content-Type": "application/json", "locale": "ko-KR"},
        )
        self._server_time_offset_ms: int = 0

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""

        await self._client.aclose()

    def _build_headers(
        self,
        *,
        method: str,
        path: str,
        query_string: str,
        body: str,
    ) -> dict[str, str]:
        """Create authenticated request headers."""

        timestamp = str(int(time.time() * 1000) + self._server_time_offset_ms)
        request_path = path
        if query_string:
            request_path = f"{path}?{query_string}"
        prehash = f"{timestamp}{method.upper()}{request_path}{body}"
        signature = base64.b64encode(
            hmac.new(
                self.api_secret.encode("utf-8"),
                prehash.encode("utf-8"),
                hashlib.sha256,
            ).digest()
        ).decode("utf-8")
        headers = {
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": self.passphrase,
        }
        if self.demo:
            headers["paptrading"] = self.settings.exchange.demo_header
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        auth: bool = False,
    ) -> dict[str, Any]:
        """Perform a REST request with retry and Bitget error handling."""

        params = {key: value for key, value in (params or {}).items() if value not in (None, "")}
        query_string = urlencode(params)
        body = json.dumps(payload or {}, separators=(",", ":"), ensure_ascii=False) if payload else ""
        headers = self._build_headers(method=method, path=path, query_string=query_string, body=body) if auth else {}

        # 공개 API에도 paptrading 헤더 추가 (DEMO 모드에서 심볼 필터링)
        if self.demo and "paptrading" not in headers:
            headers["paptrading"] = self.settings.exchange.demo_header

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.settings.exchange.max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException, BitgetAPIError)),
            reraise=True,
        ):
            with attempt:
                response = await self._client.request(
                    method,
                    path,
                    params=params,
                    json=payload,
                    headers=headers,
                )
                try:
                    data = response.json()
                except ValueError:
                    data = {}
                if response.status_code >= 400:
                    error_code = str(data.get("code") or "")
                    error_msg = str(data.get("msg") or response.text or f"HTTP {response.status_code}")
                    raise BitgetAPIError(
                        f"Bitget HTTP {response.status_code} {error_code}: {error_msg}",
                        code=error_code or None,
                        status_code=response.status_code,
                        payload=data if isinstance(data, dict) else {},
                    )
                if str(data.get("code")) != "00000":
                    error_code = str(data.get("code") or "")
                    if error_code in _TIMESTAMP_EXPIRED_CODES and auth:
                        await self._sync_server_time_offset()
                        headers = self._build_headers(
                            method=method,
                            path=path,
                            query_string=query_string,
                            body=body,
                        )
                    raise BitgetAPIError(
                        f"Bitget error {data.get('code')}: {data.get('msg')}",
                        code=error_code or None,
                        status_code=response.status_code,
                        payload=data if isinstance(data, dict) else {},
                    )
                return data
        raise BitgetAPIError("Unreachable retry state")

    async def get_server_time(self) -> int:
        """Get current server time in milliseconds."""

        data = await self._request("GET", "/api/v2/public/time")
        return int(data.get("data", {}).get("serverTime", time.time() * 1000))

    async def _sync_server_time_offset(self) -> None:
        """Fetch server time and update the local clock offset to correct timestamp drift."""

        try:
            local_before = int(time.time() * 1000)
            server_time = await self.get_server_time()
            local_after = int(time.time() * 1000)
            local_estimate = (local_before + local_after) // 2
            self._server_time_offset_ms = server_time - local_estimate
            self.logger.info(
                "Server time offset updated",
                extra={"extra_data": {"offset_ms": self._server_time_offset_ms}},
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "Failed to sync server time offset",
                extra={"extra_data": {"error": str(exc)}},
            )

    async def get_contracts(self, product_type: ProductType, symbol: str | None = None) -> list[ContractConfig]:
        """Return futures contract metadata."""

        data = await self._request(
            "GET",
            "/api/v2/mix/market/contracts",
            params={"productType": product_type.value, "symbol": symbol},
        )
        return [ContractConfig.from_api(product_type, item) for item in data.get("data", [])]

    async def get_tickers(self, product_type: ProductType) -> list[TickerSnapshot]:
        """Return all ticker snapshots for a product type."""

        data = await self._request(
            "GET",
            "/api/v2/mix/market/tickers",
            params={"productType": product_type.value},
        )
        return [TickerSnapshot.from_api(product_type, item) for item in data.get("data", [])]

    async def get_ticker(self, product_type: ProductType, symbol: str) -> TickerSnapshot | None:
        """Return a single ticker snapshot."""

        data = await self._request(
            "GET",
            "/api/v2/mix/market/ticker",
            params={"productType": product_type.value, "symbol": symbol.upper()},
        )
        items = data.get("data", [])
        if isinstance(items, dict):
            return TickerSnapshot.from_api(product_type, items)
        if items:
            return TickerSnapshot.from_api(product_type, items[0])
        return None

    async def get_candles(
        self,
        product_type: ProductType,
        symbol: str,
        granularity: str,
        *,
        limit: int = 200,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        kline_type: str = "MARKET",
    ) -> list[Candle]:
        """Get historical candles for a symbol."""

        params = {
            "productType": product_type.value,
            "symbol": symbol.upper(),
            "granularity": granularity,
            "limit": limit,
            "startTime": start_time_ms,
            "endTime": end_time_ms,
            "kLineType": kline_type,
        }
        data = await self._request("GET", "/api/v2/mix/market/candles", params=params)
        rows = data.get("data", [])
        candles = [Candle.from_api(item) for item in rows]
        return sorted(candles, key=lambda item: item.timestamp)

    async def get_merge_depth(
        self,
        product_type: ProductType,
        symbol: str,
        *,
        precision: str = "scale0",
        limit: str = "50",
    ) -> OrderBookSnapshot:
        """Return merged order book depth."""

        data = await self._request(
            "GET",
            "/api/v2/mix/market/merge-depth",
            params={
                "productType": product_type.value,
                "symbol": symbol.upper(),
                "precision": precision,
                "limit": limit,
            },
        )
        return OrderBookSnapshot.from_api(symbol, product_type, data.get("data", {}))

    async def get_symbol_prices(
        self,
        product_type: ProductType,
        symbol: str,
    ) -> dict[str, Any]:
        """Get mark and index prices."""

        data = await self._request(
            "GET",
            "/api/v2/mix/market/symbol-price",
            params={"productType": product_type.value, "symbol": symbol.upper()},
        )
        return data.get("data", {})

    async def get_current_funding_rate(
        self,
        product_type: ProductType,
        symbol: str,
    ) -> dict[str, Any]:
        """Return current funding-rate metadata."""

        data = await self._request(
            "GET",
            "/api/v2/mix/market/current-fund-rate",
            params={"productType": product_type.value, "symbol": symbol.upper()},
        )
        rows = data.get("data", [])
        return rows[0] if isinstance(rows, list) and rows else data.get("data", {})

    async def get_next_funding_time(
        self,
        product_type: ProductType,
        symbol: str,
    ) -> dict[str, Any]:
        """Return next funding settlement time."""

        data = await self._request(
            "GET",
            "/api/v2/mix/market/funding-time",
            params={"productType": product_type.value, "symbol": symbol.upper()},
        )
        rows = data.get("data", [])
        return rows[0] if isinstance(rows, list) and rows else data.get("data", {})

    async def get_account_list(self, product_type: ProductType) -> list[AccountSummary]:
        """Return account balances for a product type."""

        data = await self._request(
            "GET",
            "/api/v2/mix/account/accounts",
            params={"productType": product_type.value},
            auth=True,
        )
        return [AccountSummary.from_api(product_type, item) for item in data.get("data", [])]

    async def get_positions(
        self,
        product_type: ProductType,
        *,
        margin_coin: str | None = None,
    ) -> list[PositionSnapshot]:
        """Return open positions for a product type."""

        params = {"productType": product_type.value, "marginCoin": margin_coin}
        data = await self._request(
            "GET",
            "/api/v2/mix/position/all-position",
            params=params,
            auth=True,
        )
        return [PositionSnapshot.from_api(product_type, item) for item in data.get("data", [])]

    async def get_pending_orders(
        self,
        product_type: ProductType,
        *,
        symbol: str | None = None,
        client_order_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return pending or partially filled orders."""

        data = await self._request(
            "GET",
            "/api/v2/mix/order/orders-pending",
            params={
                "productType": product_type.value,
                "symbol": symbol.upper() if symbol else None,
                "clientOid": client_order_id,
                "status": status,
                "limit": 100,
            },
            auth=True,
        )
        raw_data = data.get("data")
        if isinstance(raw_data, dict):
            entrusted_list = raw_data.get("entrustedList")
            return entrusted_list if isinstance(entrusted_list, list) else []
        return raw_data if isinstance(raw_data, list) else []

    async def place_order(self, order: OrderRequest) -> OrderResult:
        """Place a futures order."""

        payload = order.as_api_payload()
        try:
            data = await self._request(
                "POST",
                "/api/v2/mix/order/place-order",
                payload=payload,
                auth=True,
            )
        except BitgetAPIError as exc:
            # Bitget rejects hedge-mode `tradeSide` in one-way position mode.
            if exc.code == "40774" and payload.get("tradeSide"):
                fallback_payload = dict(payload)
                fallback_payload.pop("tradeSide", None)
                self.logger.info(
                    "Retrying place-order without tradeSide for one-way position mode",
                    extra={"extra_data": {"symbol": order.symbol.upper(), "product_type": order.product_type.value}},
                )
                data = await self._request(
                    "POST",
                    "/api/v2/mix/order/place-order",
                    payload=fallback_payload,
                    auth=True,
                )
            else:
                raise
        response_data = data.get("data", {})
        return OrderResult(
            client_order_id=str(response_data.get("clientOid") or order.client_order_id or ""),
            exchange_order_id=response_data.get("orderId"),
            status=OrderStatus.NEW,
            symbol=order.symbol.upper(),
            product_type=order.product_type,
            raw=response_data,
        )

    async def modify_order(
        self,
        product_type: ProductType,
        symbol: str,
        margin_coin: str,
        order_id: str | None,
        client_order_id: str | None,
        *,
        price: float | None = None,
        quantity: float | None = None,
    ) -> dict[str, Any]:
        """Modify a live order."""

        payload = {
            "productType": product_type.value,
            "symbol": symbol.upper(),
            "marginCoin": margin_coin.upper(),
            "orderId": order_id,
            "clientOid": client_order_id,
            "newPrice": f"{price}" if price is not None else None,
            "newSize": f"{quantity}" if quantity is not None else None,
        }
        data = await self._request(
            "POST",
            "/api/v2/mix/order/modify-order",
            payload=payload,
            auth=True,
        )
        return data.get("data", {})

    async def cancel_order(
        self,
        product_type: ProductType,
        symbol: str,
        margin_coin: str,
        *,
        order_id: str | None = None,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Cancel a pending order."""

        payload = {
            "symbol": symbol.upper(),
            "productType": product_type.value,
            "marginCoin": margin_coin.upper(),
            "orderId": order_id,
            "clientOid": client_order_id,
        }
        data = await self._request(
            "POST",
            "/api/v2/mix/order/cancel-order",
            payload=payload,
            auth=True,
        )
        return data.get("data", {})

    async def place_plan_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Place a trigger order for TP/SL protection."""

        data = await self._request(
            "POST",
            "/api/v2/mix/order/place-plan-order",
            payload=payload,
            auth=True,
        )
        return data.get("data", {})

    async def get_pending_plan_orders(
        self,
        product_type: ProductType,
        *,
        plan_type: str = "profit_loss",
        symbol: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return pending trigger orders."""

        data = await self._request(
            "GET",
            "/api/v2/mix/order/orders-plan-pending",
            params={
                "productType": product_type.value,
                "planType": plan_type,
                "symbol": symbol.upper() if symbol else None,
            },
            auth=True,
        )
        if isinstance(data.get("data"), dict):
            return data["data"].get("entrustedList", [])
        return data.get("data", [])

    async def get_fills(
        self,
        product_type: ProductType,
        *,
        symbol: str | None = None,
        order_id: str | None = None,
        limit: int = 100,
    ) -> list[FillDetail]:
        """Return fill details."""

        data = await self._request(
            "GET",
            "/api/v2/mix/order/fills",
            params={
                "productType": product_type.value,
                "symbol": symbol.upper() if symbol else None,
                "orderId": order_id,
                "limit": limit,
            },
            auth=True,
        )
        rows = data.get("data", {}).get("fillList") if isinstance(data.get("data"), dict) else data.get("data", [])
        return [FillDetail.from_api(item) for item in rows or []]

    async def dry_run_validate(
        self,
        order: OrderRequest,
        contract: ContractConfig,
    ) -> list[str]:
        """Perform local order validation before sending to the exchange."""

        errors: list[str] = []
        if order.quantity <= 0:
            errors.append("주문 수량은 0보다 커야 합니다.")
        if order.order_type.value == "limit" and (order.price is None or order.price <= 0):
            errors.append("지정가 주문은 가격이 필요합니다.")
        if contract.min_order_size and order.quantity < contract.min_order_size:
            errors.append(
                f"최소 주문 수량 미달: {order.quantity} < {contract.min_order_size}"
            )
        if order.price is not None and contract.price_step:
            # 부동소수점 오차 방지: 검증 전 가격을 tick size에 맞게 정규화
            order.price = round_to_step(order.price, contract.price_step, mode="nearest")
            remainder = round(order.price / contract.price_step, 6) % 1
            if remainder > 1e-3 and remainder < (1 - 1e-3):
                errors.append("가격이 tick size 배수가 아닙니다.")
        if contract.size_step:
            remainder = round(order.quantity / contract.size_step, 12) % 1
            if remainder not in {0, 1} and remainder > 1e-8:
                errors.append("수량이 lot size 배수가 아닙니다.")
        notion = as_float(order.price or 0) * order.quantity
        if notion <= 0 and order.order_type.value == "market":
            self.logger.info("시장가 검증은 실시간 티커 가격 기준으로 보완 필요")
        return errors
