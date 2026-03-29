"""Break and retest strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from core.enums import RegimeType, Side, StrategyName
from market.indicators import atr

from .base import (
    BaseStrategy,
    SignalContext,
    StrategySignal,
    build_chart_level,
    build_chart_marker,
    build_chart_zone,
    build_entry_reason,
    humanize_reason_token,
)
from .confirmation import bearish_confirmation, bullish_confirmation, find_fvg_zone, volume_recovered
from .rr_filter import compute_structural_rr
from .structure_levels import build_structure_snapshot
from .structural_stops import detect_structural_stop
from .structural_targets import detect_structural_targets


def _candidate_levels(levels: dict[str, float], *, direction: Side) -> list[tuple[str, float]]:
    """Select relevant breakout levels for a direction."""

    if direction == Side.LONG:
        keys = ["prev_day_high", "asia_high", "london_high", "swing_high_4h", "range_high_recent", "vwap"]
    else:
        keys = ["prev_day_low", "asia_low", "london_low", "swing_low_4h", "range_low_recent", "vwap"]
    return [(name, levels[name]) for name in keys if name in levels]


def _range_middle(df_15m: pd.DataFrame, price: float, exclusion: float) -> bool:
    """Avoid range midpoint entries."""

    recent = df_15m.tail(16)
    if recent.empty:
        return False
    range_high = float(recent["high"].max())
    range_low = float(recent["low"].min())
    midpoint = (range_high + range_low) / 2
    band = (range_high - range_low) * exclusion
    return abs(price - midpoint) <= band


@dataclass(slots=True)
class BreakRetestStrategy(BaseStrategy):
    """Vincent-style break and retest with deterministic filters."""

    retest_tolerance_atr: float = 0.4
    volume_multiple: float = 1.15
    range_middle_exclusion: float = 0.2
    merge_nearby_target_threshold_pct: float = 0.0015

    name: StrategyName = StrategyName.BREAK_RETEST

    def evaluate(self, context: SignalContext) -> StrategySignal | None:
        """Evaluate long and short break-retest setups."""

        df_3m = context.frames["3m"]
        df_5m = context.frames["5m"]
        df_15m = context.frames["15m"]
        if len(df_3m) < 40 or len(df_5m) < 20 or len(df_15m) < 20:
            return None

        atr_value = float(atr(df_3m).iloc[-1])
        tq = context.trend_quality_score
        for side in [Side.LONG, Side.SHORT]:
            if side == Side.LONG and tq < -0.3:
                continue
            if side == Side.SHORT and tq > 0.3:
                continue
            signal = self._evaluate_side(context, df_3m, df_5m, df_15m, atr_value, side)
            if signal:
                return signal
        return None

    def _evaluate_side(
        self,
        context: SignalContext,
        df_3m: pd.DataFrame,
        df_5m: pd.DataFrame,
        df_15m: pd.DataFrame,
        atr_value: float,
        side: Side,
    ) -> StrategySignal | None:
        """Evaluate a single direction."""

        last_close = float(df_3m["close"].iloc[-1])
        last_open = float(df_3m["open"].iloc[-1])
        last_low = float(df_3m["low"].iloc[-1])
        last_high = float(df_3m["high"].iloc[-1])
        prior_close = float(df_3m["close"].iloc[-2])
        structure = build_structure_snapshot(df_entry=df_3m, df_structure=df_15m)

        if _range_middle(df_15m, last_close, self.range_middle_exclusion):
            return None

        for level_name, level_value in _candidate_levels(context.levels, direction=side):
            broke = (
                side == Side.LONG
                and prior_close <= level_value
                and last_close > level_value
            ) or (
                side == Side.SHORT
                and prior_close >= level_value
                and last_close < level_value
            )
            if not broke:
                continue

            chase_distance = abs(last_close - level_value)
            if chase_distance > atr_value * 0.8:
                continue

            fvg_zone = find_fvg_zone(df_3m.tail(5))
            retest_zone_hit = (
                abs(last_low - level_value) <= atr_value * self.retest_tolerance_atr
                if side == Side.LONG
                else abs(last_high - level_value) <= atr_value * self.retest_tolerance_atr
            )
            if fvg_zone is not None:
                low_zone, high_zone = min(fvg_zone), max(fvg_zone)
                if side == Side.LONG:
                    retest_zone_hit = retest_zone_hit or low_zone <= last_low <= high_zone
                else:
                    retest_zone_hit = retest_zone_hit or low_zone <= last_high <= high_zone

            if not retest_zone_hit:
                continue
            # 3m confirmation만으로 충분 (5m은 선택)
            if side == Side.LONG and not bullish_confirmation(df_3m.tail(3)):
                continue
            if side == Side.SHORT and not bearish_confirmation(df_3m.tail(3)):
                continue
            if not volume_recovered(df_3m, self.volume_multiple):
                continue

            entry = last_close
            setup_context = {
                "side": side,
                "entry_price": entry,
                "atr": atr_value,
                "breakout_level": level_value,
                "range_boundaries": structure["range_boundaries"],
                "recent_swings_entry": structure["recent_swings_entry"],
                "recent_swings_structure": structure["recent_swings_structure"],
                "retest_low": min(last_low, level_value) if side == Side.LONG else None,
                "retest_high": max(last_high, level_value) if side == Side.SHORT else None,
                "fvg_zone": fvg_zone,
                # 패턴 무효화: 리테스트 레벨 (브레이크 포인트)
                "pattern_invalidation_low": min(last_low, level_value) if side == Side.LONG else None,
                "pattern_invalidation_high": max(last_high, level_value) if side == Side.SHORT else None,
            }
            stop, stop_reason, stop_meta = detect_structural_stop(signal_context=context, setup_context=setup_context)
            if stop is None or stop_reason is None:
                continue
            targets = detect_structural_targets(
                context,
                setup_context,
                merge_threshold_pct=self.merge_nearby_target_threshold_pct,
                range_middle_exclusion=self.range_middle_exclusion,
            )
            if len(targets) < 1:
                continue
            tp1 = float(targets[0]["price"])
            tp2 = float(targets[1]["price"]) if len(targets) > 1 else None
            tp3 = float(targets[2]["price"]) if len(targets) > 2 else None
            expected_r = compute_structural_rr(entry, stop, tp2 or tp1, side)
            if expected_r <= 0:
                continue
            confidence = 0.62 if context.regime == RegimeType.TRENDING else 0.54
            score = min(0.95, confidence + (0.08 if volume_recovered(df_5m, 1.0) else 0.0))
            retest_band = atr_value * self.retest_tolerance_atr
            zone_low = level_value - retest_band
            zone_high = level_value + retest_band
            if fvg_zone is not None:
                zone_low = min(zone_low, min(fvg_zone))
                zone_high = max(zone_high, max(fvg_zone))
            entry_reason = build_entry_reason(
                title=f"{humanize_reason_token(level_name)} 돌파 후 재지지 확인" if side == Side.LONG else f"{humanize_reason_token(level_name)} 이탈 후 재저항 확인",
                lines=[
                    (
                        f"{humanize_reason_token(level_name)} {level_value:,.4f} 를 {'위로 돌파' if side == Side.LONG else '아래로 이탈'}한 뒤 "
                        f"다시 같은 구간을 {'지지' if side == Side.LONG else '저항'}으로 확인했습니다."
                    ),
                    (
                        f"리테스트 존 {zone_low:,.4f} ~ {zone_high:,.4f} 에서 "
                        f"{'매수 우위' if side == Side.LONG else '매도 우위'} 확인 캔들이 나왔습니다."
                    ),
                    "3분과 5분 기준 확인 캔들과 거래량 회복이 동시에 붙었습니다.",
                ],
                chart_levels=[
                    build_chart_level("돌파 기준선", level_value, color="#f59e0b"),
                ],
                chart_zones=[
                    build_chart_zone("리테스트 존", zone_low, zone_high, color="#38bdf8", alpha=0.10),
                ],
                chart_marker=build_chart_marker(
                    "확인 캔들",
                    entry,
                    candle_time=df_3m.index[-1].isoformat(),
                    color="#22c55e" if side == Side.LONG else "#ef4444",
                ),
            )
            # 지정가 진입: 핵심 레벨에서 대기
            _optimal = level_value
            if side == Side.SHORT and _optimal <= last_close:
                _optimal = None  # 현재가보다 낮으면 지정가 무의미
            elif side == Side.LONG and _optimal >= last_close:
                _optimal = None  # 현재가보다 높으면 지정가 무의미

            return StrategySignal(
                symbol=context.symbol,
                product_type=context.product_type,
                strategy=self.name,
                side=side,
                entry_price=entry,
                stop_price=stop,
                tp1_price=tp1,
                tp2_price=tp2,
                tp3_price=tp3,
                optimal_entry_price=_optimal,
                score=score,
                confidence=confidence,
                expected_r=expected_r,
                fees_r=0.0,
                slippage_r=0.0,
                display_name=self.name.display_name,
                regime=context.regime,
                stop_reason=stop_reason,
                target_plan=targets,
                rr_to_tp1=compute_structural_rr(entry, stop, tp1, side),
                rr_to_tp2=compute_structural_rr(entry, stop, tp2, side),
                rr_to_best_target=max(compute_structural_rr(entry, stop, item["price"], side) for item in targets),
                tags=["break", "retest", level_name],
                rationale={
                    **entry_reason,
                    "level": level_name,
                    "level_price": level_value,
                    "atr": atr_value,
                    "fvg_zone": fvg_zone,
                    "regime": context.regime.value,
                    "stop_meta": stop_meta,
                    "target_plan": targets,
                    "news_blocked": context.blocked_by_news,
                    "last_open": last_open,
                    "last_close": last_close,
                },
            )
        return None
