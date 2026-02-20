from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class OrderIntent:
    symbol: str
    side: str
    qty: float
    price: Optional[float] = None
    order_type: str = "market"
    tif: str = "GTC"


def build_order_intent(symbol: str, side: str, qty: float, price: Optional[float] = None, order_type: str = "market") -> OrderIntent:
    return OrderIntent(
        symbol=str(symbol or "").strip(),
        side=str(side or "").strip().lower(),
        qty=float(qty or 0.0),
        price=(float(price) if price is not None else None),
        order_type=str(order_type or "market").strip().lower(),
    )


def intent_to_dict(intent: OrderIntent) -> Dict[str, Any]:
    return {
        "symbol": intent.symbol,
        "side": intent.side,
        "qty": float(intent.qty),
        "price": (float(intent.price) if intent.price is not None else None),
        "order_type": intent.order_type,
        "tif": intent.tif,
    }
