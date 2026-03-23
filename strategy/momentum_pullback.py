"""Momentum pullback / flag continuation strategy."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.enums import RegimeType, Side, StrategyName
from market.indicators import atr, vwap

from .base import (
    BaseStrategy,
    SignalContext,
    StrategySignal,
    build_chart_level,
    build_chart_marker,
    build_chart_zone,
    build_entry_reason,
)
from .confirmation import bearish_confirmation, bullish_confirmation, volume_recovered
from .rr_filter import compute_structural_rr
from .structure_levels import build_structure_snapshot
from .structural_stops import detect_structural_stop
from .structural_targets import detect_structural_targets


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Return an EMA series."""

    return series.ewm(span=span, adjust=False).mean()


@dataclass(slots=True)
class MomentumPullbackStrategy(BaseStrategy):
    """Trade the first shallow pullback after a strong impulse move."""

    volume_multiple: float = 1.1
    max_pullback_ratio: float = 0.45
    merge_nearby_target_threshold_pct: float = 0.0015
    name: StrategyName = StrategyName.MOMENTUM_PULLBACK

    def evaluate(self, context: SignalContext) -> StrategySignal | None:
        """Return a continuation setup after a strong impulse and controlled pullback."""

        df_3m = context.frames["3m"]
        df_5m = context.frames["5m"]
        df_15m = context.frames["15m"]
        if len(df_3m) < 80 or len(df_5m) < 40 or len(df_15m) < 24:
            return None

        atr_value = float(atr(df_3m).iloc[-1])
        if atr_value <= 0:
            return None
        ema20 = _ema(df_3m["close"], 20)
        ema50 = _ema(df_3m["close"], 50)
        vwap_value = float(vwap(df_3m).iloc[-1])
        structure = build_structure_snapshot(df_entry=df_3m, df_structure=df_15m)

        tq = context.trend_quality_score
        for side in [Side.LONG, Side.SHORT]:
            if side == Side.LONG and tq < -0.3:
                continue
            if side == Side.SHORT and tq > 0.3:
                continue
            signal = self._evaluate_side(
                context=context,
                df_3m=df_3m,
                atr_value=atr_value,
                ema20=ema20,
                ema50=ema50,
                vwap_value=vwap_value,
                structure=structure,
                side=side,
            )
            if signal is not None:
                return signal
        return None

    def _evaluate_side(
        self,
        *,
        context: SignalContext,
        df_3m: pd.DataFrame,
        atr_value: float,
        ema20: pd.Series,
        ema50: pd.Series,
        vwap_value: float,
        structure: dict[str, dict[str, float]],
        side: Side,
    ) -> StrategySignal | None:
        """Evaluate one direction of the momentum-pullback setup."""

        last_close = float(df_3m["close"].iloc[-1])
        last_low = float(df_3m["low"].iloc[-1])
        last_high = float(df_3m["high"].iloc[-1])
        last_ema20 = float(ema20.iloc[-1])
        last_ema50 = float(ema50.iloc[-1])

        impulse_window = df_3m.tail(12)
        if side == Side.LONG:
            impulse_low = float(impulse_window["low"].min())
            impulse_high = float(impulse_window["high"].max())
            impulse_size = impulse_high - impulse_low
            pullback_low = float(df_3m.tail(5)["low"].min())
            trigger_level = float(df_3m.tail(4)["high"].iloc[:-1].max())
            pullback_ratio = (impulse_high - pullback_low) / max(impulse_size, 1e-9)
            valid_trend = last_close > last_ema20 > last_ema50 and last_close >= vwap_value
            valid_trigger = bullish_confirmation(df_3m.tail(3)) and last_close > trigger_level
            stop_hint_low = pullback_low
        else:
            impulse_high = float(impulse_window["high"].max())
            impulse_low = float(impulse_window["low"].min())
            impulse_size = impulse_high - impulse_low
            pullback_high = float(df_3m.tail(5)["high"].max())
            trigger_level = float(df_3m.tail(4)["low"].iloc[:-1].min())
            pullback_ratio = (pullback_high - impulse_low) / max(impulse_size, 1e-9)
            valid_trend = last_close < last_ema20 < last_ema50 and last_close <= vwap_value
            valid_trigger = bearish_confirmation(df_3m.tail(3)) and last_close < trigger_level
            stop_hint_low = pullback_high

        if impulse_size < atr_value * 1.5:
            return None
        if pullback_ratio <= 0 or pullback_ratio > self.max_pullback_ratio:
            return None
        if not valid_trend or not valid_trigger or not volume_recovered(df_3m, self.volume_multiple):
            return None

        setup_context = {
            "side": side,
            "entry_price": last_close,
            "atr": atr_value,
            "breakout_level": trigger_level,
            "range_boundaries": structure["range_boundaries"],
            "recent_swings_entry": structure["recent_swings_entry"],
            "recent_swings_structure": structure["recent_swings_structure"],
            "retest_low": stop_hint_low if side == Side.LONG else None,
            "retest_high": stop_hint_low if side == Side.SHORT else None,
            # 패턴 무효화: 모멘텀 트리거 레벨
            "pattern_invalidation_low": stop_hint_low if side == Side.LONG else None,
            "pattern_invalidation_high": trigger_level if side == Side.SHORT else None,
        }
        stop, stop_reason, stop_meta = detect_structural_stop(signal_context=context, setup_context=setup_context)
        if stop is None or stop_reason is None:
            return None
        targets = detect_structural_targets(
            context,
            setup_context,
            merge_threshold_pct=self.merge_nearby_target_threshold_pct,
            range_middle_exclusion=0.15,
        )
        if len(targets) < 1:
            return None
        tp1 = float(targets[0]["price"])
        tp2 = float(targets[1]["price"]) if len(targets) > 1 else None
        tp3 = float(targets[2]["price"]) if len(targets) > 2 else None
        rr_to_tp2 = compute_structural_rr(last_close, stop, tp2 or tp1, side)
        if rr_to_tp2 <= 0:
            return None

        confidence = 0.66 if context.regime in {RegimeType.TRENDING, RegimeType.EXPANSION} else 0.56
        pullback_zone_low = min(trigger_level, stop_hint_low)
        pullback_zone_high = max(trigger_level, stop_hint_low)
        entry_reason = build_entry_reason(
            title="강한 확장 뒤 첫 눌림 재개",
            lines=[
                f"직전 확장폭이 ATR 대비 {impulse_size / max(atr_value, 1e-9):.2f}배로 강한 impulse 가 먼저 나왔습니다.",
                f"눌림 깊이는 전체 impulse 의 {pullback_ratio * 100:.1f}% 수준으로 얕게 유지됐습니다.",
                f"EMA20·VWAP 구조를 유지한 채 trigger {trigger_level:,.4f} 를 {'재돌파' if side == Side.LONG else '재이탈'}하며 재개 진입했습니다.",
            ],
            chart_levels=[
                build_chart_level("트리거 레벨", trigger_level, color="#f59e0b"),
                build_chart_level("EMA20 기준", last_ema20, color="#38bdf8"),
                build_chart_level("VWAP 기준", vwap_value, color="#a78bfa", linestyle="-." ),
            ],
            chart_zones=[
                build_chart_zone("눌림 구간", pullback_zone_low, pullback_zone_high, color="#14b8a6", alpha=0.10),
            ],
            chart_marker=build_chart_marker(
                "재개 캔들",
                last_close,
                candle_time=df_3m.index[-1].isoformat(),
                color="#22c55e" if side == Side.LONG else "#ef4444",
            ),
        )
        return StrategySignal(
            symbol=context.symbol,
            product_type=context.product_type,
            strategy=self.name,
            side=side,
            entry_price=last_close,
            stop_price=stop,
            tp1_price=tp1,
            tp2_price=tp2,
            tp3_price=tp3,
            score=min(0.94, confidence + 0.04),
            confidence=confidence,
            expected_r=rr_to_tp2,
            fees_r=0.0,
            slippage_r=0.0,
            display_name=self.name.display_name,
            regime=context.regime,
            stop_reason=stop_reason,
            target_plan=targets,
            rr_to_tp1=compute_structural_rr(last_close, stop, tp1, side),
            rr_to_tp2=rr_to_tp2,
            rr_to_best_target=max(compute_structural_rr(last_close, stop, item["price"], side) for item in targets),
            tags=["momentum_pullback", "flag", "continuation"],
            rationale={
                **entry_reason,
                "impulse_size": impulse_size,
                "pullback_ratio": pullback_ratio,
                "trigger_level": trigger_level,
                "ema20": last_ema20,
                "ema50": last_ema50,
                "vwap": vwap_value,
                "atr": atr_value,
                "stop_meta": stop_meta,
                "target_plan": targets,
                "regime": context.regime.value,
            },
        )
