"""Market regime classification."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from core.enums import RegimeType, StrategyName

from .indicators import adx, atr, rolling_percentile, swing_points, trend_quality, vwap


@dataclass(slots=True)
class RegimeSnapshot:
    """Output of regime classification."""

    regime: RegimeType
    adx_value: float
    atr_percentile: float
    volume_percentile: float
    trend_quality_score: float
    above_vwap: bool
    compression_ratio: float = 1.0
    vwap_distance_pct: float = 0.0
    swing_quality_score: float = 0.0
    session_context: str = "mixed"
    allowed_strategies: list[StrategyName] = field(default_factory=list)
    penalty_flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class MarketRegimeClassifier:
    """Classify the market into trade-routing regimes."""

    def allowed_strategies(self, regime: RegimeType) -> list[StrategyName]:
        """Return the strategies enabled for a regime."""

        mapping = {
            RegimeType.TRENDING: [StrategyName.BREAK_RETEST, StrategyName.CHOCH, StrategyName.FAIR_VALUE_GAP],
            RegimeType.RANGING: [StrategyName.LIQUIDITY_RAID, StrategyName.ORDER_BLOCK],
            RegimeType.EXPANSION: [StrategyName.BREAK_RETEST, StrategyName.FAIR_VALUE_GAP, StrategyName.ORDER_BLOCK],
            RegimeType.EVENT_RISK: [],
            RegimeType.DEAD_MARKET: [],
            RegimeType.UNKNOWN: [StrategyName.BREAK_RETEST, StrategyName.LIQUIDITY_RAID, StrategyName.FAIR_VALUE_GAP],
        }
        return mapping.get(regime, [])

    def _compression_ratio(self, df_15m: pd.DataFrame) -> float:
        """Measure recent range compression versus a larger lookback."""

        recent = df_15m.tail(12)
        wider = df_15m.tail(48)
        if recent.empty or wider.empty:
            return 1.0
        recent_range = max(float(recent["high"].max() - recent["low"].min()), 1e-9)
        wider_range = max(float(wider["high"].max() - wider["low"].min()), 1e-9)
        return max(0.0, min(3.0, recent_range / wider_range))

    def _swing_quality(self, df_1h: pd.DataFrame) -> float:
        """Estimate how cleanly higher-timeframe swings are progressing."""

        swings = swing_points(df_1h.tail(60), window=2)
        highs = [price for _, price in swings["highs"][-4:]]
        lows = [price for _, price in swings["lows"][-4:]]
        quality = 0.0
        if len(highs) >= 2 and highs[-1] > highs[-2]:
            quality += 0.5
        if len(lows) >= 2 and lows[-1] > lows[-2]:
            quality += 0.5
        if len(highs) >= 2 and highs[-1] < highs[-2]:
            quality += 0.5
        if len(lows) >= 2 and lows[-1] < lows[-2]:
            quality += 0.5
        return min(1.0, quality)

    def _session_context(self, df_15m: pd.DataFrame) -> str:
        """Return a rough session bucket from the latest candle."""

        if df_15m.empty:
            return "mixed"
        hour = int(df_15m.index[-1].hour)
        if 0 <= hour < 8:
            return "asia"
        if 7 <= hour < 16:
            return "london"
        if 13 <= hour < 22:
            return "newyork"
        return "mixed"

    def classify(self, df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> RegimeSnapshot:
        """Classify a symbol using multi-timeframe structure."""

        if df_15m.empty or df_1h.empty:
            empty_regime = RegimeType.UNKNOWN
            return RegimeSnapshot(
                regime=empty_regime,
                adx_value=0.0,
                atr_percentile=0.0,
                volume_percentile=0.0,
                trend_quality_score=0.0,
                above_vwap=False,
                compression_ratio=1.0,
                vwap_distance_pct=0.0,
                swing_quality_score=0.0,
                session_context="mixed",
                allowed_strategies=self.allowed_strategies(empty_regime),
                notes=["시계열 데이터 부족"],
            )

        adx_value = float(adx(df_15m).iloc[-1])
        atr_percentile = float(rolling_percentile(atr(df_15m), window=100).iloc[-1])
        volume_percentile = float(rolling_percentile(df_15m["volume"], window=100).iloc[-1])
        tq = float(trend_quality(df_1h["close"], window=20).iloc[-1])
        vwap_series = vwap(df_15m)
        last_close = float(df_15m["close"].iloc[-1])
        vwap_value = float(vwap_series.iloc[-1])
        above_vwap = last_close >= vwap_value
        vwap_distance_pct = abs(last_close - vwap_value) / max(abs(vwap_value), 1e-9)
        compression_ratio = self._compression_ratio(df_15m)
        swing_quality = self._swing_quality(df_1h)
        session_context = self._session_context(df_15m)
        bar_ranges = (df_15m["high"] - df_15m["low"]).tail(48)
        latest_bar_range = float(bar_ranges.iloc[-1]) if not bar_ranges.empty else 0.0
        baseline_bar_range = float(bar_ranges.median()) if not bar_ranges.empty else 0.0
        shock_ratio = latest_bar_range / max(baseline_bar_range, 1e-9)
        latest_volume = float(df_15m["volume"].iloc[-1])
        median_volume = float(df_15m["volume"].tail(48).median())
        volume_spike_ratio = latest_volume / max(median_volume, 1e-9)
        recent_range_pct = (
            float(df_15m.tail(24)["high"].max() - df_15m.tail(24)["low"].min())
            / max(abs(last_close), 1e-9)
        )
        penalty_flags: list[str] = []
        notes: list[str] = []

        if recent_range_pct <= 0.006 and vwap_distance_pct <= 0.0015 and compression_ratio <= 0.8:
            notes.append("절대 변동폭이 너무 작음")
            penalty_flags.append("dead_market")
            regime = RegimeType.DEAD_MARKET
        elif shock_ratio >= 1.8 and volume_spike_ratio >= 1.8 and atr_percentile > 0.75:
            notes.append("이벤트/뉴스성 급변 가능성")
            penalty_flags.append("event_risk")
            regime = RegimeType.EVENT_RISK
        elif adx_value <= 14 and atr_percentile <= 0.25 and volume_percentile <= 0.3:
            notes.append("거래량·변동성 모두 부족")
            penalty_flags.append("dead_market")
            regime = RegimeType.DEAD_MARKET
        elif compression_ratio <= 0.45 and atr_percentile >= 0.7 and volume_percentile >= 0.6 and recent_range_pct >= 0.01:
            notes.append("압축 이후 확장 구간")
            regime = RegimeType.EXPANSION
        elif adx_value >= 24 and abs(tq) >= 0.45 and vwap_distance_pct >= 0.0015:
            notes.append("구조적 추세 확인")
            regime = RegimeType.TRENDING
        elif adx_value <= 18 and atr_percentile <= 0.55:
            notes.append("저변동 횡보")
            regime = RegimeType.RANGING
        else:
            regime = RegimeType.UNKNOWN
            notes.append("혼합 구간")

        if above_vwap:
            notes.append("VWAP 상단")
        else:
            notes.append("VWAP 하단")
        if compression_ratio <= 0.5:
            notes.append("최근 압축")
        if session_context != "mixed":
            notes.append(f"{session_context} 세션 문맥")

        return RegimeSnapshot(
            regime=regime,
            adx_value=adx_value,
            atr_percentile=atr_percentile,
            volume_percentile=volume_percentile,
            trend_quality_score=tq,
            above_vwap=above_vwap,
            compression_ratio=compression_ratio,
            vwap_distance_pct=vwap_distance_pct,
            swing_quality_score=swing_quality,
            session_context=session_context,
            allowed_strategies=self.allowed_strategies(regime),
            penalty_flags=penalty_flags,
            notes=notes,
        )
