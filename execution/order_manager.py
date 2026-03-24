"""Order idempotency, submission, and synchronization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from core.enums import OrderType
from core.logger import get_logger
from core.persistence import SQLitePersistence
from core.settings import AppSettings
from core.state_store import StateStore
from core.utils import build_client_order_id
from exchange.bitget_demo import BitgetExchangeBase
from exchange.bitget_models import ContractConfig, OrderRequest, OrderResult
from risk.risk_engine import ApprovedTrade


@dataclass(slots=True)
class PendingOrder:
    """Runtime pending order record."""

    client_order_id: str
    symbol: str
    created_at: datetime
    route_notes: list[str]


class OrderManager:
    """Handle order submission, sync, and deduplication."""

    def __init__(
        self,
        settings: AppSettings,
        exchange: BitgetExchangeBase,
        persistence: SQLitePersistence,
        state_store: StateStore,
    ) -> None:
        self.settings = settings
        self.exchange = exchange
        self.persistence = persistence
        self.state_store = state_store
        self.logger = get_logger(__name__)
        self._pending_by_signal: dict[str, PendingOrder] = {}

    async def restore_open_orders(self) -> list[dict[str, Any]]:
        """Recover pending orders after restart."""

        recovered = await self.exchange.get_pending_orders()
        for order in recovered:
            client_oid = str(order.get("clientOid", ""))
            if not client_oid:
                continue
            self.state_store.update_order(client_oid, order)
        return recovered

    async def submit_entry(
        self,
        approved_trade: ApprovedTrade,
        contract: ContractConfig,
        order: OrderRequest,
        route_notes: list[str],
    ) -> OrderResult:
        """Validate, deduplicate, and place an entry order."""

        signal = approved_trade.signal
        existing = self._pending_by_signal.get(signal.signal_id)
        if existing:
            raise RuntimeError(f"중복 주문 방지: {signal.signal_id}")

        client_oid = build_client_order_id("entry", signal.symbol, signal.strategy.value, signal.side.value, signal.signal_id[:6])
        order.client_order_id = client_oid
        validation_errors = await self.exchange.rest.dry_run_validate(order, contract)
        if validation_errors:
            raise ValueError("; ".join(validation_errors))

        result = await self.exchange.place_order(order)
        now_iso = datetime.now(tz=UTC).isoformat(timespec="seconds")
        order_payload = {
            "client_order_id": result.client_order_id,
            "exchange_order_id": result.exchange_order_id,
            "symbol": signal.symbol,
            "mode": self.exchange.mode.value,
            "side": signal.side.value,
            "order_type": order.order_type.value,
            "status": result.status.value,
            "price": order.price,
            "quantity": order.quantity,
            "filled_quantity": 0.0,
            "avg_fill_price": None,
            "reduce_only": order.reduce_only,
            "reason": ",".join(route_notes),
            "signal_id": signal.signal_id,
            "strategy": signal.strategy.value,
            "display_name": signal.display_name or signal.strategy.display_name,
            "stop_price": approved_trade.stop_price,
            "tp1_price": approved_trade.tp1_price,
            "tp2_price": approved_trade.tp2_price,
            "tp3_price": approved_trade.tp3_price,
            "tags": signal.tags,
            "rationale": signal.rationale,
            "stop_reason": approved_trade.stop_reason,
            "target_plan": approved_trade.target_plan,
            "rr_to_tp1": approved_trade.rr_to_tp1,
            "rr_to_tp2": approved_trade.rr_to_tp2,
            "rr_to_best_target": approved_trade.rr_to_best_target,
            "trade_rejected_reason": approved_trade.trade_rejected_reason,
            "ev_metrics": approved_trade.ev_metrics,
            "market_regime": signal.regime.value,
            "chosen_strategy": signal.chosen_strategy or signal.strategy.value,
            "candidate_strategies": signal.candidate_strategies,
            "rejected_strategies": signal.rejected_strategies,
            "rejection_reasons": signal.rejection_reasons,
            "overlap_score": signal.overlap_score,
            "conflict_resolution_decision": signal.conflict_resolution_decision,
            "leverage": approved_trade.leverage,
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        self.persistence.save_order(order_payload)
        self.state_store.update_order(client_oid, order_payload)
        self._pending_by_signal[signal.signal_id] = PendingOrder(
            client_order_id=client_oid,
            symbol=signal.symbol,
            created_at=datetime.now(tz=UTC),
            route_notes=route_notes,
        )
        return result

    async def cancel_if_stale(
        self,
        *,
        contract: ContractConfig,
        signal_id: str,
        margin_coin: str,
        max_age_seconds: int | None = None,
    ) -> bool:
        """Cancel a maker order if it has been pending too long."""

        max_age = max_age_seconds or self.settings.execution.maker_timeout_seconds
        pending = self._pending_by_signal.get(signal_id)
        if not pending:
            return False
        if datetime.now(tz=UTC) - pending.created_at < timedelta(seconds=max_age):
            return False
        try:
            await self.exchange.rest.cancel_order(
                contract.product_type,
                contract.symbol,
                margin_coin,
                client_order_id=pending.client_order_id,
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "Cancel stale order failed — keeping in state",
                extra={"extra_data": {"symbol": contract.symbol, "signal_id": signal_id, "error": str(exc)}},
            )
            return False
        self._pending_by_signal.pop(signal_id, None)
        self.state_store.remove_order(pending.client_order_id)
        return True

    async def sync_pending_orders(self) -> list[dict[str, Any]]:
        """Refresh pending orders from the exchange."""

        current = await self.exchange.get_pending_orders()
        live_ids = {str(item.get("clientOid", "")) for item in current if item.get("clientOid")}

        # Detect orphaned orders: in local state but not on exchange
        local_ids = set(self.state_store.state.open_orders.keys())
        orphaned = local_ids - live_ids
        for orphan_id in orphaned:
            orphan = self.state_store.state.open_orders.get(orphan_id, {})
            self.logger.warning(
                "Orphaned order removed from local state",
                extra={"extra_data": {"client_order_id": orphan_id, "symbol": orphan.get("symbol")}},
            )
            self.state_store.remove_order(orphan_id)
            # Also clean up signal tracking
            signal_id = str(orphan.get("signal_id") or "")
            if signal_id:
                self._pending_by_signal.pop(signal_id, None)

        for item in current:
            client_oid = str(item.get("clientOid", ""))
            if not client_oid:
                continue
            existing = self.state_store.state.open_orders.get(client_oid, {})
            # ✅ side는 기존 "long"/"short" 값을 유지 (exchange "buy"/"sell"로 덮지 않음)
            existing_side = existing.get("side", "")
            exchange_side = item.get("side", "")
            normalized_side = existing_side if existing_side in ("long", "short") else (
                "long" if exchange_side in ("buy", "open_long") else
                "short" if exchange_side in ("sell", "open_short") else exchange_side
            )
            payload = {
                **existing,
                "client_order_id": client_oid,
                "exchange_order_id": item.get("orderId"),
                "symbol": item.get("symbol"),
                "mode": self.exchange.mode.value,
                "side": normalized_side,
                "order_type": item.get("orderType", existing.get("order_type", "")),
                "status": item.get("state", item.get("status", "open")),
                "price": item.get("price") or existing.get("price"),
                "quantity": item.get("size") or existing.get("quantity"),
                "filled_quantity": item.get("baseVolume"),
                "avg_fill_price": item.get("priceAvg") or None,
                "reduce_only": str(item.get("reduceOnly", "")).lower() == "yes",
                "reason": "exchange_sync",
                "created_at": existing.get("created_at", datetime.now(tz=UTC).isoformat(timespec="seconds")),
                "updated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            }
            self.persistence.save_order(payload)
            self.state_store.update_order(client_oid, payload)
        return current

    async def close_position_market(
        self,
        *,
        symbol: str,
        product_type: Any,
        margin_coin: str,
        side: str,
        quantity: float,
        close_reason: str = "manual_close",
    ) -> OrderResult:
        """Issue a reduce-only market order."""

        request = OrderRequest(
            symbol=symbol,
            product_type=product_type,
            margin_coin=margin_coin,
            side="sell" if side == "long" else "buy",
            trade_side=None,  # one-way 모드에서는 trade_side 미사용
            order_type=OrderType.MARKET,
            quantity=quantity,
            reduce_only=True,
        )
        result = await self.exchange.place_order(request)

        # ✅ NEW: Save close order to persistence with reduce_only flag
        now_iso = datetime.now(tz=UTC).isoformat(timespec="seconds")
        close_order_payload = {
            "client_order_id": result.client_order_id,
            "exchange_order_id": result.exchange_order_id,
            "symbol": symbol,
            "mode": self.exchange.mode.value,
            "side": side,
            "order_type": OrderType.MARKET.value,
            "status": result.status.value,
            "price": None,
            "quantity": quantity,
            "filled_quantity": 0.0,
            "avg_fill_price": None,
            "reduce_only": True,
            "reason": close_reason,
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        self.persistence.save_order(close_order_payload)
        self.state_store.update_order(result.client_order_id, close_order_payload)
        self.logger.info(
            "Close order submitted and persisted",
            extra={"extra_data": {"symbol": symbol, "quantity": quantity, "client_order_id": result.client_order_id}}
        )
        return result

    async def replace_stale_limit_with_market(
        self,
        *,
        contract: ContractConfig,
        order_payload: dict[str, Any],
    ) -> OrderResult | None:
        """Replace a stale entry limit order with a market order fallback."""

        if str(order_payload.get("order_type") or "") != OrderType.LIMIT.value:
            return None

        created_at_raw = str(order_payload.get("created_at") or "")
        if not created_at_raw:
            return None
        try:
            created_at = datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
        except ValueError:
            return None
        if datetime.now(tz=UTC) - created_at < timedelta(seconds=self.settings.execution.maker_timeout_seconds):
            return None

        client_oid = str(order_payload.get("client_order_id") or "")
        if not client_oid:
            return None

        try:
            await self.exchange.rest.cancel_order(
                contract.product_type,
                contract.symbol,
                contract.margin_coin,
                client_order_id=client_oid,
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "Failed to cancel stale entry limit",
                extra={"extra_data": {"symbol": contract.symbol, "client_order_id": client_oid, "error": str(exc)}},
            )
            return None

        signal_id = str(order_payload.get("signal_id") or "")
        if signal_id:
            self._pending_by_signal.pop(signal_id, None)
        self.state_store.remove_order(client_oid)

        # taker_fallback=true 이면 시장가 전환, false 이면 취소만
        if not self.settings.execution.taker_fallback:
            self.logger.info(
                "지정가 미체결 취소 (핵심 레벨 도달 실패)",
                extra={"extra_data": {"symbol": contract.symbol, "client_order_id": client_oid}},
            )
            return None

        fallback_order = OrderRequest(
            symbol=contract.symbol,
            product_type=contract.product_type,
            margin_coin=contract.margin_coin,
            side="buy" if order_payload.get("side") == "long" else "sell",
            trade_side="open",
            order_type=OrderType.MARKET,
            quantity=float(order_payload.get("quantity") or 0.0),
            preset_stop_loss_price=order_payload.get("stop_price") if self.settings.execution.use_exchange_plan_orders else None,
        )
        fallback_order.client_order_id = build_client_order_id(
            "entryfb",
            contract.symbol,
            str(order_payload.get("strategy") or "unknown"),
            str(order_payload.get("side") or "long"),
            signal_id[:6] if signal_id else "manual",
        )
        validation_errors = await self.exchange.rest.dry_run_validate(fallback_order, contract)
        if validation_errors:
            raise ValueError("; ".join(validation_errors))

        result = await self.exchange.place_order(fallback_order)
        now_iso = datetime.now(tz=UTC).isoformat(timespec="seconds")
        replacement_payload = {
            **order_payload,
            "client_order_id": result.client_order_id,
            "exchange_order_id": result.exchange_order_id,
            "order_type": OrderType.MARKET.value,
            "status": result.status.value,
            "reason": "maker_timeout_fallback",
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        self.persistence.save_order(replacement_payload)
        self.state_store.update_order(result.client_order_id, replacement_payload)
        return result
