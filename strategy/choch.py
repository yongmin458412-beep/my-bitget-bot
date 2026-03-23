"""Change of Character (CHoCH) 전략 — 추세 전환 초기 진입."""

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
from .confirmation import bearish_confirmation, bullish_confirmation, mss_bearish, mss_bullish
from .rr_filter import compute_structural_rr
from .structural_stops import detect_structural_stop
from .structural_targets import detect_structural_targets
from .structure_levels import build_structure_snapshot


@dataclass(slots=True)
class CHoCHStrategy(BaseStrategy):
    """시장 구조 전환(CHoCH) 이후 리테스트 진입.

    불리시 CHoCH: 하락 구조에서 이전 스윙 고점을 돌파 → 추세 전환 신호
    베어리시 CHoCH: 상승 구조에서 이전 스윙 저점을 돌파 → 추세 전환 신호
    CHoCH 레벨 리테스트 시 진입.
    """

    lookback_swings: int = 20            # 스윙 탐색 범위 (캔들 수)
    min_swing_atr_multiple: float = 1.5  # 스윙 고저 최소 크기
    max_retest_distance_atr: float = 0.5 # 리테스트 허용 거리
    merge_nearby_target_threshold_pct: float = 0.0015
    name: StrategyName = StrategyName.CHOCH

    def evaluate(self, context: SignalContext) -> StrategySignal | None:
        """CHoCH 진입 신호를 반환."""

        df_3m = context.frames.get("3m")
        df_15m = context.frames.get("15m")
        if df_3m is None or df_15m is None:
            return None
        if len(df_3m) < 80 or len(df_15m) < 30:
            return None

        # 횡보/데드 마켓에서는 CHoCH 진입 제외
        if context.regime in {RegimeType.RANGING, RegimeType.DEAD_MARKET}:
            return None

        # 추세 방향 필터: 하락추세(-0.3 이하)면 LONG 제외, 상승추세(+0.3 이상)면 SHORT 제외
        tq = context.trend_quality_score

        atr_value = float(atr(df_3m).iloc[-1])
        if atr_value <= 0:
            return None

        for side in (Side.LONG, Side.SHORT):
            # 추세 역방향 진입 차단: 하락추세에서 LONG, 상승추세에서 SHORT 금지
            if side == Side.LONG and tq < -0.3:
                continue
            if side == Side.SHORT and tq > 0.3:
                continue
            signal = self._scan_choch(context, df_3m, df_15m, atr_value, side)
            if signal is not None:
                return signal
        return None

    def _scan_choch(
        self,
        context: SignalContext,
        df_3m: pd.DataFrame,
        df_15m: pd.DataFrame,
        atr_value: float,
        side: Side,
    ) -> StrategySignal | None:
        """CHoCH 레벨 감지 → 리테스트 확인 → 신호 반환."""

        last_close = float(df_3m["close"].iloc[-1])
        window = df_3m.tail(self.lookback_swings + 5)

        if side == Side.LONG:
            # 불리시 CHoCH: 최근 구조에서 스윙 고점 돌파
            # 1. 구조 하락 확인: 이전 스윙 고점 식별
            swing_highs = []
            for j in range(2, len(window) - 2):
                h = float(window["high"].iloc[j])
                if h > float(window["high"].iloc[j - 1]) and h > float(window["high"].iloc[j + 1]):
                    swing_highs.append(h)
            if len(swing_highs) < 2:
                return None

            # 가장 최근 스윙 고점이 CHoCH 레벨
            choch_level = swing_highs[-1]
            prev_swing_high = swing_highs[-2]

            # CHoCH 조건: 현재 CHoCH 레벨이 이전보다 낮아야 (하락 구조)
            if choch_level >= prev_swing_high:
                return None

            # 현재 가격이 CHoCH 레벨을 최근에 돌파하고 리테스트 중
            breakout_occurred = last_close > choch_level
            retest_in_range = abs(last_close - choch_level) < atr_value * self.max_retest_distance_atr

            if not breakout_occurred or not retest_in_range:
                return None

            # MSS 확인 (구조 전환 미니 신호)
            if not mss_bullish(df_3m.tail(8)):
                return None

            confirmed = bullish_confirmation(df_3m.tail(2))

        else:
            # 베어리시 CHoCH: 최근 구조에서 스윙 저점 이탈
            swing_lows = []
            for j in range(2, len(window) - 2):
                l = float(window["low"].iloc[j])
                if l < float(window["low"].iloc[j - 1]) and l < float(window["low"].iloc[j + 1]):
                    swing_lows.append(l)
            if len(swing_lows) < 2:
                return None

            choch_level = swing_lows[-1]
            prev_swing_low = swing_lows[-2]

            # CHoCH 조건: 현재 CHoCH 레벨이 이전보다 높아야 (상승 구조)
            if choch_level <= prev_swing_low:
                return None

            breakout_occurred = last_close < choch_level
            retest_in_range = abs(last_close - choch_level) < atr_value * self.max_retest_distance_atr

            if not breakout_occurred or not retest_in_range:
                return None

            if not mss_bearish(df_3m.tail(8)):
                return None

            confirmed = bearish_confirmation(df_3m.tail(2))

        if not confirmed:
            return None

        # CHoCH 레벨이 ATR 최소 크기 이상인지 확인
        if abs(choch_level - last_close) < atr_value * 0.1:
            return None

        return self._build_signal(
            context=context,
            df_3m=df_3m,
            df_15m=df_15m,
            atr_value=atr_value,
            side=side,
            choch_level=choch_level,
        )

    def _build_signal(
        self,
        *,
        context: SignalContext,
        df_3m: pd.DataFrame,
        df_15m: pd.DataFrame,
        atr_value: float,
        side: Side,
        choch_level: float,
    ) -> StrategySignal | None:
        last_close = float(df_3m["close"].iloc[-1])
        structure = build_structure_snapshot(df_entry=df_3m, df_structure=df_15m)

        setup_context: dict = {
            "side": side,
            "entry_price": last_close,
            "atr": atr_value,
            "breakout_level": choch_level,
            "range_boundaries": structure["range_boundaries"],
            "recent_swings_entry": structure["recent_swings_entry"],
            "recent_swings_structure": structure["recent_swings_structure"],
            "retest_low": choch_level if side == Side.LONG else None,
            "retest_high": choch_level if side == Side.SHORT else None,
            # 패턴 무효화: CHoCH 구조 전환 레벨
            "pattern_invalidation_low": choch_level if side == Side.LONG else None,
            "pattern_invalidation_high": choch_level if side == Side.SHORT else None,
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

        # CHoCH 는 추세전환이라 신뢰도 보수적으로
        confidence = 0.68 if context.regime in {RegimeType.TRENDING, RegimeType.EXPANSION} else 0.58

        dist_pct = abs(last_close - choch_level) / max(choch_level, 1e-9) * 100

        entry_reason = build_entry_reason(
            title=f"CHoCH 추세전환 진입 ({'불리시' if side == Side.LONG else '베어리시'})",
            lines=[
                f"CHoCH 레벨 {choch_level:,.4f} 돌파 후 리테스트 ({dist_pct:.2f}% 거리).",
                f"시장 구조 전환 확인 — {'하락→상승' if side == Side.LONG else '상승→하락'} CHoCH.",
                f"확인 캔들 완성, 손절 {stop:,.4f} ({stop_reason})",
            ],
            chart_levels=[
                build_chart_level("CHoCH 레벨", choch_level, color="#ec4899"),
            ],
            chart_zones=[
                build_chart_zone(
                    "CHoCH 리테스트 구간",
                    choch_level - atr_value * 0.3,
                    choch_level + atr_value * 0.3,
                    color="#ec4899",
                    alpha=0.12,
                ),
            ],
            chart_marker=build_chart_marker(
                "CHoCH 진입",
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
            score=min(0.92, confidence + 0.04),
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
            tags=["choch", "change_of_character", "bos", "smc", "reversal"],
            rationale={
                **entry_reason,
                "choch_level": choch_level,
                "retest_distance_pct": dist_pct,
                "atr": atr_value,
                "stop_meta": stop_meta,
                "target_plan": targets,
                "regime": context.regime.value,
            },
        )
