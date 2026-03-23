"""Order Block (OB) 전략 — 기관 대량주문 구간 재진입."""

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
)
from .confirmation import bearish_confirmation, bullish_confirmation, volume_recovered
from .rr_filter import compute_structural_rr
from .structural_stops import detect_structural_stop
from .structural_targets import detect_structural_targets
from .structure_levels import build_structure_snapshot


@dataclass(slots=True)
class OrderBlockStrategy(BaseStrategy):
    """강한 임펄스 이전의 마지막 역방향 캔들(OB)로 되돌아올 때 진입.

    불리시 OB: 강한 상승 직전 마지막 음봉 구간
    베어리시 OB: 강한 하락 직전 마지막 양봉 구간
    가격이 OB 하위 50% 에 진입 시 반응 진입.
    """

    min_impulse_atr_multiple: float = 1.8   # 임펄스 최소 크기 (ATR 배수)
    max_ob_age_candles: int = 60             # OB 최대 유효 캔들 수
    ob_entry_zone_ratio: float = 0.65        # OB 구간 진입 비율 (0.65 = 하위 65%)
    min_ob_body_ratio: float = 0.3           # OB 캔들 몸통 비율 최소값
    merge_nearby_target_threshold_pct: float = 0.0015
    name: StrategyName = StrategyName.ORDER_BLOCK

    def evaluate(self, context: SignalContext) -> StrategySignal | None:
        """OB 진입 신호를 반환."""

        df_3m = context.frames.get("3m")
        df_15m = context.frames.get("15m")
        if df_3m is None or df_15m is None:
            return None
        if len(df_3m) < 80 or len(df_15m) < 20:
            return None

        atr_value = float(atr(df_3m).iloc[-1])
        if atr_value <= 0:
            return None

        for side in (Side.LONG, Side.SHORT):
            signal = self._scan_ob(context, df_3m, df_15m, atr_value, side)
            if signal is not None:
                return signal
        return None

    def _scan_ob(
        self,
        context: SignalContext,
        df_3m: pd.DataFrame,
        df_15m: pd.DataFrame,
        atr_value: float,
        side: Side,
    ) -> StrategySignal | None:
        """OB 스캔 후 현재 가격이 OB 구간에 있으면 신호 반환."""

        last_close = float(df_3m["close"].iloc[-1])
        lookback = min(self.max_ob_age_candles, len(df_3m) - 5)

        for i in range(5, lookback):
            ob_candle = df_3m.iloc[-(i + 1)]
            impulse_slice = df_3m.iloc[-i:-1]

            ob_open = float(ob_candle["open"])
            ob_close = float(ob_candle["close"])
            ob_high = float(ob_candle["high"])
            ob_low = float(ob_candle["low"])
            ob_body = abs(ob_close - ob_open)
            ob_range = ob_high - ob_low

            if ob_range <= 0 or ob_body / ob_range < self.min_ob_body_ratio:
                continue

            if side == Side.LONG:
                # 불리시 OB: 음봉 (ob_close < ob_open)
                if ob_close >= ob_open:
                    continue
                impulse_move = float(impulse_slice["close"].iloc[-1]) - float(impulse_slice["open"].iloc[0])
                if impulse_move < atr_value * self.min_impulse_atr_multiple:
                    continue
                # OB 구간: ob_low ~ ob_high, 하위 50% 진입
                ob_zone_top = ob_low + (ob_high - ob_low) * self.ob_entry_zone_ratio
                price_in_ob = ob_low <= last_close <= ob_zone_top

            else:
                # 베어리시 OB: 양봉 (ob_close > ob_open)
                if ob_close <= ob_open:
                    continue
                impulse_move = float(impulse_slice["open"].iloc[0]) - float(impulse_slice["close"].iloc[-1])
                if impulse_move < atr_value * self.min_impulse_atr_multiple:
                    continue
                # OB 구간: ob_low ~ ob_high, 상위 50% 진입
                ob_zone_bottom = ob_high - (ob_high - ob_low) * self.ob_entry_zone_ratio
                price_in_ob = ob_zone_bottom <= last_close <= ob_high

            if not price_in_ob:
                continue

            # 확인 캔들 + 거래량 체크
            confirmed = (
                bullish_confirmation(df_3m.tail(2)) if side == Side.LONG
                else bearish_confirmation(df_3m.tail(2))
            )
            if not confirmed:
                continue
            if not volume_recovered(df_3m, 1.05):
                continue

            ob_zone_low = ob_low
            ob_zone_high = ob_high

            return self._build_signal(
                context=context,
                df_3m=df_3m,
                df_15m=df_15m,
                atr_value=atr_value,
                side=side,
                ob_low=ob_zone_low,
                ob_high=ob_zone_high,
                ob_open=ob_open,
                ob_close=ob_close,
            )

        return None

    def _build_signal(
        self,
        *,
        context: SignalContext,
        df_3m: pd.DataFrame,
        df_15m: pd.DataFrame,
        atr_value: float,
        side: Side,
        ob_low: float,
        ob_high: float,
        ob_open: float,
        ob_close: float,
    ) -> StrategySignal | None:
        last_close = float(df_3m["close"].iloc[-1])
        structure = build_structure_snapshot(df_entry=df_3m, df_structure=df_15m)

        setup_context: dict = {
            "side": side,
            "entry_price": last_close,
            "atr": atr_value,
            "breakout_level": ob_high if side == Side.LONG else ob_low,
            "range_boundaries": structure["range_boundaries"],
            "recent_swings_entry": structure["recent_swings_entry"],
            "recent_swings_structure": structure["recent_swings_structure"],
            "retest_low": ob_low if side == Side.LONG else None,
            "retest_high": ob_high if side == Side.SHORT else None,
            # 패턴 무효화: 오더블록 경계 (매물대 고점/저점)
            "pattern_invalidation_low": ob_low if side == Side.LONG else None,
            "pattern_invalidation_high": ob_high if side == Side.SHORT else None,
        }
        stop, stop_reason, stop_meta = detect_structural_stop(
            signal_context=context, setup_context=setup_context
        )
        if stop is None or stop_reason is None:
            return None

        targets = detect_structural_targets(
            context,
            setup_context,
            merge_threshold_pct=self.merge_nearby_target_threshold_pct,
            range_middle_exclusion=0.1,
        )
        if len(targets) < 1:
            return None

        tp1 = float(targets[0]["price"])
        tp2 = float(targets[1]["price"]) if len(targets) > 1 else None
        tp3 = float(targets[2]["price"]) if len(targets) > 2 else None
        rr_to_tp1 = compute_structural_rr(last_close, stop, tp1, side)
        rr_to_tp2 = compute_structural_rr(last_close, stop, tp2 or tp1, side)
        if rr_to_tp1 <= 0:
            return None

        confidence = 0.70 if context.regime in {RegimeType.TRENDING, RegimeType.EXPANSION} else 0.60

        entry_reason = build_entry_reason(
            title=f"오더 블록 진입 ({'불리시' if side == Side.LONG else '베어리시'} OB)",
            lines=[
                f"OB 구간 {ob_low:,.4f} ~ {ob_high:,.4f} — 기관 대량 주문 흔적.",
                f"{'음봉' if side == Side.LONG else '양봉'} OB (O:{ob_open:.4f} / C:{ob_close:.4f}) 후 강한 임펄스 확인.",
                f"현재 가격 {last_close:,.4f} 이 OB 진입 구간에서 확인 캔들 완성.",
            ],
            chart_levels=[
                build_chart_level("OB 상단", ob_high, color="#8b5cf6"),
                build_chart_level("OB 하단", ob_low, color="#8b5cf6", linestyle=":"),
            ],
            chart_zones=[
                build_chart_zone("오더 블록", ob_low, ob_high, color="#8b5cf6", alpha=0.15),
            ],
            chart_marker=build_chart_marker(
                "OB 진입",
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
            rr_to_tp1=rr_to_tp1,
            rr_to_tp2=rr_to_tp2,
            rr_to_best_target=max(
                compute_structural_rr(last_close, stop, item["price"], side) for item in targets
            ),
            tags=["order_block", "ob", "smc", "institutional"],
            rationale={
                **entry_reason,
                "ob_low": ob_low,
                "ob_high": ob_high,
                "ob_open": ob_open,
                "ob_close": ob_close,
                "atr": atr_value,
                "stop_meta": stop_meta,
                "target_plan": targets,
                "regime": context.regime.value,
            },
        )
