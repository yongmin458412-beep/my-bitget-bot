"""Session range breakout strategy."""

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


def _active_session(levels: dict[str, float], latest_hour: int) -> tuple[str, float | None, float | None]:
    """Return the session range most relevant to the latest timestamp."""

    if 0 <= latest_hour < 8 and levels.get("asia_high") is not None and levels.get("asia_low") is not None:
        return "asia", levels.get("asia_high"), levels.get("asia_low")
    if 7 <= latest_hour < 16 and levels.get("london_high") is not None and levels.get("london_low") is not None:
        return "london", levels.get("london_high"), levels.get("london_low")
    if levels.get("london_high") is not None and levels.get("london_low") is not None:
        return "london", levels.get("london_high"), levels.get("london_low")
    return "asia", levels.get("asia_high"), levels.get("asia_low")


@dataclass(slots=True)
class SessionBreakoutStrategy(BaseStrategy):
    """Trade compressed session ranges that break with expanding participation."""

    volume_multiple: float = 1.2
    max_session_width_atr: float = 6.0
    merge_nearby_target_threshold_pct: float = 0.0015
    name: StrategyName = StrategyName.SESSION_BREAKOUT

    def evaluate(self, context: SignalContext) -> StrategySignal | None:
        """Return a session-breakout signal when compression expands into a breakout."""

        df_3m = context.frames["3m"]
        df_5m = context.frames["5m"]
        df_15m = context.frames["15m"]
        if len(df_3m) < 60 or len(df_5m) < 30 or len(df_15m) < 24:
            return None

        atr_value = float(atr(df_3m).iloc[-1])
        if atr_value <= 0:
            return None
        session_name, session_high, session_low = _active_session(context.levels, int(df_15m.index[-1].hour))
        if session_high is None or session_low is None:
            return None
        session_width = float(session_high - session_low)
        if session_width <= 0 or session_width > atr_value * self.max_session_width_atr:
            return None

        last_close = float(df_3m["close"].iloc[-1])
        previous_close = float(df_3m["close"].iloc[-2])
        last_low = float(df_3m["low"].iloc[-1])
        last_high = float(df_3m["high"].iloc[-1])
        vwap_value = float(vwap(df_3m).iloc[-1])
        structure = build_structure_snapshot(df_entry=df_3m, df_structure=df_15m)

        for side in [Side.LONG, Side.SHORT]:
            breakout_level = session_high if side == Side.LONG else session_low
            broke = (
                side == Side.LONG
                and previous_close <= breakout_level
                and last_close > breakout_level
                and last_close >= vwap_value
                and bullish_confirmation(df_3m.tail(3))
            ) or (
                side == Side.SHORT
                and previous_close >= breakout_level
                and last_close < breakout_level
                and last_close <= vwap_value
                and bearish_confirmation(df_3m.tail(3))
            )
            if not broke or not volume_recovered(df_3m, self.volume_multiple):
                continue

            setup_context = {
                "side": side,
                "entry_price": last_close,
                "atr": atr_value,
                "breakout_level": breakout_level,
                "range_boundaries": structure["range_boundaries"],
                "recent_swings_entry": structure["recent_swings_entry"],
                "recent_swings_structure": structure["recent_swings_structure"],
                "retest_low": min(last_low, breakout_level) if side == Side.LONG else None,
                "retest_high": max(last_high, breakout_level) if side == Side.SHORT else None,
                # 패턴 무효화: 세션 레인지 경계
                "pattern_invalidation_low": session_low if side == Side.LONG else None,
                "pattern_invalidation_high": session_high if side == Side.SHORT else None,
            }
            stop, stop_reason, stop_meta = detect_structural_stop(signal_context=context, setup_context=setup_context)
            if stop is None or stop_reason is None:
                continue
            targets = detect_structural_targets(
                context,
                setup_context,
                merge_threshold_pct=self.merge_nearby_target_threshold_pct,
                range_middle_exclusion=0.15,
            )
            if len(targets) < 1:
                continue

            tp1 = float(targets[0]["price"])
            tp2 = float(targets[1]["price"]) if len(targets) > 1 else None
            tp3 = float(targets[2]["price"]) if len(targets) > 2 else None
            rr_to_tp2 = compute_structural_rr(last_close, stop, tp2 or tp1, side)
            if rr_to_tp2 <= 0:
                continue

            confidence = 0.68 if context.regime == RegimeType.EXPANSION else 0.58
            entry_reason = build_entry_reason(
                title=f"{session_name.upper()} 세션 레인지 {'상단' if side == Side.LONG else '하단'} 돌파",
                lines=[
                    f"{session_name.upper()} 세션 레인지 {session_low:,.4f} ~ {session_high:,.4f} 가 먼저 압축됐습니다.",
                    f"이후 거래량이 붙으면서 레인지 {'상단' if side == Side.LONG else '하단'} {breakout_level:,.4f} 을 {'돌파' if side == Side.LONG else '이탈'}했습니다.",
                    f"가격이 VWAP {'위' if side == Side.LONG else '아래'}를 유지한 채 확인 캔들이 이어져 추세 확장으로 진입했습니다.",
                ],
                chart_levels=[
                    build_chart_level("세션 상단", session_high, color="#f59e0b"),
                    build_chart_level("세션 하단", session_low, color="#60a5fa"),
                    build_chart_level("실제 돌파선", breakout_level, color="#22c55e" if side == Side.LONG else "#ef4444", linestyle="-." ),
                ],
                chart_zones=[
                    build_chart_zone("세션 레인지", session_low, session_high, color="#a78bfa", alpha=0.08),
                ],
                chart_marker=build_chart_marker(
                    "돌파 캔들",
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
                score=min(0.95, confidence + 0.06),
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
                tags=["session_breakout", session_name, "volume_expansion"],
                rationale={
                    **entry_reason,
                    "session_name": session_name,
                    "session_high": session_high,
                    "session_low": session_low,
                    "session_width": session_width,
                    "vwap": vwap_value,
                    "atr": atr_value,
                    "stop_meta": stop_meta,
                    "target_plan": targets,
                    "regime": context.regime.value,
                },
            )
        return None
