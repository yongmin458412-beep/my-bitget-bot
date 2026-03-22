"""Liquidity raid reversal strategy."""

from __future__ import annotations

from dataclasses import dataclass

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
from .confirmation import bearish_confirmation, bullish_confirmation, mss_bearish, mss_bullish, volume_recovered
from .rr_filter import compute_structural_rr
from .structure_levels import build_structure_snapshot
from .structural_stops import detect_structural_stop
from .structural_targets import detect_structural_targets


def _sweep_candidates(context: SignalContext, side: Side) -> list[tuple[str, float]]:
    """Return sweep targets."""

    if side == Side.LONG:
        levels = [
            ("asia_low", context.levels.get("asia_low")),
            ("london_low", context.levels.get("london_low")),
            ("prev_day_low", context.levels.get("prev_day_low")),
        ]
        levels.extend([("equal_lows", value) for value in context.level_tags.get("equal_lows", [])])
    else:
        levels = [
            ("asia_high", context.levels.get("asia_high")),
            ("london_high", context.levels.get("london_high")),
            ("prev_day_high", context.levels.get("prev_day_high")),
        ]
        levels.extend([("equal_highs", value) for value in context.level_tags.get("equal_highs", [])])
    return [(name, value) for name, value in levels if value is not None]


@dataclass(slots=True)
class LiquidityRaidStrategy(BaseStrategy):
    """JadeCap-style liquidity raid reversal."""

    volume_multiple: float = 1.1
    range_middle_exclusion: float = 0.2
    merge_nearby_target_threshold_pct: float = 0.0015
    name: StrategyName = StrategyName.LIQUIDITY_RAID

    def evaluate(self, context: SignalContext) -> StrategySignal | None:
        """Evaluate long and short raid setups."""

        df_3m = context.frames["3m"]
        df_5m = context.frames["5m"]
        df_15m = context.frames["15m"]
        if len(df_3m) < 50 or len(df_5m) < 25 or len(df_15m) < 20:
            return None

        atr_value = float(atr(df_3m).iloc[-1])
        for side in [Side.LONG, Side.SHORT]:
            signal = self._evaluate_side(context, df_3m, df_5m, atr_value, side)
            if signal:
                return signal
        return None

    def _evaluate_side(
        self,
        context: SignalContext,
        df_3m: pd.DataFrame,
        df_5m: pd.DataFrame,
        atr_value: float,
        side: Side,
    ) -> StrategySignal | None:
        """Evaluate one direction of the raid strategy."""

        last = df_3m.iloc[-1]
        previous = df_3m.iloc[-2]
        structure = build_structure_snapshot(df_entry=df_3m, df_structure=context.frames["15m"])
        for level_name, level_price in _sweep_candidates(context, side):
            if side == Side.LONG:
                swept = float(previous["low"]) < level_price and float(last["close"]) > level_price
                confirm = bullish_confirmation(df_3m.tail(3)) and mss_bullish(df_5m.tail(8))
            else:
                swept = float(previous["high"]) > level_price and float(last["close"]) < level_price
                confirm = bearish_confirmation(df_3m.tail(3)) and mss_bearish(df_5m.tail(8))
            if not swept or not confirm:
                continue
            if not volume_recovered(df_3m, self.volume_multiple):
                continue

            entry = float(last["close"])
            setup_context = {
                "side": side,
                "entry_price": entry,
                "atr": atr_value,
                "sweep_low": float(previous["low"]) if side == Side.LONG else None,
                "sweep_high": float(previous["high"]) if side == Side.SHORT else None,
                "range_boundaries": structure["range_boundaries"],
                "recent_swings_entry": structure["recent_swings_entry"],
                "recent_swings_structure": structure["recent_swings_structure"],
                "reclaim_level": level_price,
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
            confidence = 0.64 if context.regime in {RegimeType.RANGING, RegimeType.EVENT_RISK} else 0.55
            reclaim_zone = (
                (level_price - atr_value * 0.15, level_price + atr_value * 0.15)
                if side == Side.SHORT
                else (level_price - atr_value * 0.15, level_price + atr_value * 0.15)
            )
            entry_reason = build_entry_reason(
                title=f"{humanize_reason_token(level_name)} 스윕 후 재진입 반전",
                lines=[
                    f"{humanize_reason_token(level_name)} {level_price:,.4f} 를 먼저 {'하향 스윕' if side == Side.LONG else '상향 스윕'}했습니다.",
                    f"이후 가격이 다시 레벨 {'위' if side == Side.LONG else '아래'}로 복귀하며 reclaim 이 확인됐습니다.",
                    f"5분 구조 전환(MSS)과 {'상승' if side == Side.LONG else '하락'} 확인 캔들이 붙어 반전 진입했습니다.",
                ],
                chart_levels=[
                    build_chart_level("스윕 기준선", level_price, color="#f97316"),
                ],
                chart_zones=[
                    build_chart_zone("재진입 존", reclaim_zone[0], reclaim_zone[1], color="#14b8a6", alpha=0.10),
                ],
                chart_marker=build_chart_marker(
                    "Reclaim",
                    entry,
                    candle_time=df_3m.index[-1].isoformat(),
                    color="#22c55e" if side == Side.LONG else "#ef4444",
                ),
            )
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
                score=min(0.94, confidence + 0.05),
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
                tags=["raid", "reclaim", level_name],
                rationale={
                    **entry_reason,
                    "liquidity_level": level_price,
                    "atr": atr_value,
                    "regime": context.regime.value,
                    "stop_meta": stop_meta,
                    "target_plan": targets,
                    "last_close": float(last["close"]),
                },
            )
        return None
