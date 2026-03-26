"""Shared enumerations used across the trading system."""

from __future__ import annotations

from enum import Enum


class TradingMode(str, Enum):
    """Runtime trading mode."""

    DEMO = "DEMO"
    LIVE = "LIVE"


class ProductType(str, Enum):
    """Supported Bitget futures market families."""

    USDT_FUTURES = "USDT-FUTURES"
    USDC_FUTURES = "USDC-FUTURES"
    COIN_FUTURES = "COIN-FUTURES"


class Side(str, Enum):
    """Position direction."""

    LONG = "long"
    SHORT = "short"


class OrderSide(str, Enum):
    """Exchange order side."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Supported order types."""

    LIMIT = "limit"
    MARKET = "market"


class TimeInForce(str, Enum):
    """Time in force policies used by the router."""

    GTC = "gtc"
    POST_ONLY = "post_only"
    IOC = "ioc"
    FOK = "fok"


class OrderStatus(str, Enum):
    """Normalized order states."""

    NEW = "new"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    FAILED = "failed"


class PositionStatus(str, Enum):
    """Normalized position states."""

    OPEN = "open"
    CLOSED = "closed"
    UNKNOWN = "unknown"


class StrategyName(str, Enum):
    """Strategy identifiers."""

    BREAK_RETEST = "break_retest"
    LIQUIDITY_RAID = "liquidity_raid"
    FAIR_VALUE_GAP = "fair_value_gap"
    ORDER_BLOCK = "order_block"
    CHOCH = "choch"
    MANUAL_DEMO = "manual_demo"

    @property
    def display_name(self) -> str:
        """Return a human-readable Korean label."""

        labels = {
            StrategyName.BREAK_RETEST: "브레이크 리테스트",
            StrategyName.LIQUIDITY_RAID: "유동성 리클레임",
            StrategyName.FAIR_VALUE_GAP: "공정가치 갭 (FVG)",
            StrategyName.ORDER_BLOCK: "오더 블록 (OB)",
            StrategyName.CHOCH: "추세전환 (CHoCH)",
            StrategyName.MANUAL_DEMO: "수동 데모",
        }
        return labels.get(self, self.value)


class RegimeType(str, Enum):
    """High-level market regime classifications."""

    TRENDING = "trend"
    TREND = "trend"
    RANGING = "range"
    RANGE = "range"
    EXPANSION = "expansion"
    EVENT_RISK = "event_risk"
    NEWS_VOLATILITY = "event_risk"
    DEAD_MARKET = "dead_market"
    RISK_OFF = "dead_market"
    UNKNOWN = "unknown"

    @property
    def display_name(self) -> str:
        """Return a human-readable Korean label."""

        labels = {
            RegimeType.TRENDING: "추세장",
            RegimeType.RANGING: "횡보장",
            RegimeType.EXPANSION: "확장장",
            RegimeType.EVENT_RISK: "이벤트 위험 구간",
            RegimeType.DEAD_MARKET: "죽은 장",
            RegimeType.UNKNOWN: "혼합 구간",
        }
        return labels.get(self, self.value)


class ImpactLevel(str, Enum):
    """News or event impact level."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EventType(str, Enum):
    """Normalized event category."""

    NEWS = "news"
    ECONOMIC = "economic"
    EXCHANGE_NOTICE = "exchange_notice"
    SYSTEM = "system"


class SignalStatus(str, Enum):
    """Signal lifecycle state."""

    DETECTED = "detected"
    APPROVED = "approved"
    BLOCKED = "blocked"
    EXECUTED = "executed"
    EXPIRED = "expired"


class BotStatus(str, Enum):
    """Top-level bot runtime state."""

    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
