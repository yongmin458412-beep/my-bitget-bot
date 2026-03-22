"""Base types for rule-based strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from core.enums import RegimeType, Side, StrategyName
from core.utils import hash_text


_HUMAN_TERM_MAP = {
    "prev_day_high": "전일 고점",
    "prev_day_low": "전일 저점",
    "asia_high": "아시아 고점",
    "asia_low": "아시아 저점",
    "london_high": "런던 고점",
    "london_low": "런던 저점",
    "swing_high_4h": "4시간 스윙 고점",
    "swing_low_4h": "4시간 스윙 저점",
    "range_high_recent": "최근 박스 상단",
    "range_low_recent": "최근 박스 하단",
    "range_opposite_side": "레인지 반대편",
    "equal_highs": "이퀄 하이",
    "equal_lows": "이퀄 로우",
    "previous_high": "전고점",
    "previous_low": "전저점",
    "recent_swing_high": "최근 스윙 고점",
    "recent_swing_low": "최근 스윙 저점",
    "higher_timeframe_swing_high": "상위 스윙 고점",
    "higher_timeframe_swing_low": "상위 스윙 저점",
    "session_high": "세션 고점",
    "session_low": "세션 저점",
    "vwap": "VWAP",
}


@dataclass(slots=True)
class SignalContext:
    """Shared context passed to strategies."""

    symbol: str
    product_type: str
    frames: dict[str, pd.DataFrame]
    levels: dict[str, float]
    level_tags: dict[str, list[float]]
    regime: RegimeType
    regime_notes: list[str]
    ticker: dict[str, Any]
    orderbook: dict[str, Any]
    historical_stats: dict[str, float]
    blocked_by_news: bool = False
    news_penalty: float = 0.0
    trend_quality_score: float = 0.0  # 양수=상승추세, 음수=하락추세


@dataclass(slots=True)
class StrategySignal:
    """Raw strategy signal before EV/risk validation."""

    symbol: str
    product_type: str
    strategy: StrategyName
    side: Side
    entry_price: float
    stop_price: float
    tp1_price: float
    tp2_price: float
    score: float
    confidence: float
    expected_r: float
    fees_r: float
    slippage_r: float
    tp3_price: float | None = None
    stop_reason: str = ""
    target_plan: list[dict[str, Any]] = field(default_factory=list)
    rr_to_tp1: float = 0.0
    rr_to_tp2: float = 0.0
    rr_to_best_target: float = 0.0
    display_name: str = ""
    regime: RegimeType = RegimeType.UNKNOWN
    candidate_strategies: list[str] = field(default_factory=list)
    chosen_strategy: str = ""
    rejected_strategies: list[str] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)
    overlap_score: float = 0.0
    conflict_resolution_decision: str = ""
    trade_rejected_reason: str | None = None
    ev_metrics: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    rationale: dict[str, Any] = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat(timespec="seconds"))

    @property
    def signal_id(self) -> str:
        """Create a stable-ish signal id for journaling."""

        raw = f"{self.symbol}|{self.strategy.value}|{self.side.value}|{self.created_at}|{self.entry_price:.8f}|{self.stop_price:.8f}"
        return hash_text(raw)[:24]

    @property
    def risk_per_unit(self) -> float:
        """Absolute price distance to the stop."""

        return abs(self.entry_price - self.stop_price)


class BaseStrategy:
    """Strategy protocol."""

    name: StrategyName

    def evaluate(self, context: SignalContext) -> StrategySignal | None:
        """Return a signal if the setup is present."""

        raise NotImplementedError


def humanize_reason_token(value: str | None) -> str:
    """Convert internal strategy tokens into Korean-friendly labels."""

    raw = str(value or "").strip()
    if not raw:
        return "-"
    return _HUMAN_TERM_MAP.get(raw, raw.replace("_", " "))


def build_chart_level(label: str, price: float, *, color: str, linestyle: str = "--") -> dict[str, Any]:
    """Build a chart level annotation payload."""

    return {
        "label": label,
        "price": float(price),
        "color": color,
        "linestyle": linestyle,
    }


def build_chart_zone(label: str, low: float, high: float, *, color: str, alpha: float = 0.12) -> dict[str, Any]:
    """Build a chart zone annotation payload."""

    return {
        "label": label,
        "low": float(min(low, high)),
        "high": float(max(low, high)),
        "color": color,
        "alpha": alpha,
    }


def build_chart_marker(
    label: str,
    price: float,
    *,
    candle_time: Any,
    color: str,
    marker: str = "o",
) -> dict[str, Any]:
    """Build a chart marker payload."""

    return {
        "label": label,
        "price": float(price),
        "candle_time": candle_time,
        "color": color,
        "marker": marker,
    }


def build_entry_reason(
    *,
    title: str,
    lines: list[str],
    chart_levels: list[dict[str, Any]] | None = None,
    chart_zones: list[dict[str, Any]] | None = None,
    chart_marker: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a normalized strategy rationale block for chart/text rendering."""

    return {
        "entry_reason_title": title,
        "entry_reason_lines": [line for line in lines if str(line).strip()][:4],
        "chart_levels": chart_levels or [],
        "chart_zones": chart_zones or [],
        "chart_marker": chart_marker or {},
    }
