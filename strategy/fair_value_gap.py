"""Fair Value Gap (FVG) 전략 — 기관 공백 구간 재진입."""

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
from .confirmation import bearish_confirmation, bullish_confirmation
from .rr_filter import compute_structural_rr
from .structural_stops import detect_structural_stop
from .structural_targets import detect_structural_targets
from .structure_levels import build_structure_snapshot


@dataclass(slots=True)
class FairValueGapStrategy(BaseStrategy):
    """가격이 급하게 움직일 때 남긴 FVG 구간으로 되돌아올 때 진입.

    3-캔들 패턴:
      - 불리시 FVG: candle[n-2].high < candle[n].low  (아래 공백)
      - 베어리시 FVG: candle[n-2].low > candle[n].high (위 공백)
    가격이 FVG 중간선(50%) 에 진입하면 반응 진입.
    """

    min_fvg_atr_multiple: float = 0.15   # FVG 크기 최소 조건 (ATR 의 0.15배)
    max_fvg_age_candles: int = 60        # FVG 최대 유효 캔들 수 (6시간)
    entry_fill_ratio: float = 0.35       # FVG 진입 깊이 (0.35 = 35% 수준)
    merge_nearby_target_threshold_pct: float = 0.0015
    name: StrategyName = StrategyName.FAIR_VALUE_GAP

    def evaluate(self, context: SignalContext) -> StrategySignal | None:
        """FVG 진입 신호를 반환."""

        df_3m = context.frames.get("3m")
        df_15m = context.frames.get("15m")
        if df_3m is None or df_15m is None:
            return None
        if len(df_3m) < 60 or len(df_15m) < 20:
            return None

        atr_value = float(atr(df_3m).iloc[-1])
        if atr_value <= 0:
            return None

        tq = context.trend_quality_score
        for side in (Side.LONG, Side.SHORT):
            if side == Side.LONG and tq < -0.3:
                continue
            if side == Side.SHORT and tq > 0.3:
                continue
            signal = self._scan_fvg(context, df_3m, df_15m, atr_value, side)
            if signal is not None:
                return signal
        return None

    def _scan_fvg(
        self,
        context: SignalContext,
        df_3m: pd.DataFrame,
        df_15m: pd.DataFrame,
        atr_value: float,
        side: Side,
    ) -> StrategySignal | None:
        """최근 캔들에서 유효한 FVG 를 스캔하고 진입 신호를 반환."""

        last_close = float(df_3m["close"].iloc[-1])
        lookback = min(self.max_fvg_age_candles, len(df_3m) - 3)

        for i in range(lookback, 0, -1):
            idx = -(i + 2)
            left = df_3m.iloc[idx]
            right = df_3m.iloc[idx + 2]
            fvg_low: float
            fvg_high: float

            if side == Side.LONG:
                # 불리시 FVG: 왼쪽 캔들 상단 < 오른쪽 캔들 하단
                fvg_low = float(left["high"])
                fvg_high = float(right["low"])
                if fvg_high <= fvg_low:
                    continue
                midpoint = fvg_low + (fvg_high - fvg_low) * self.entry_fill_ratio
                price_in_gap = fvg_low <= last_close <= midpoint
            else:
                # 베어리시 FVG: 왼쪽 캔들 하단 > 오른쪽 캔들 상단
                fvg_high = float(left["low"])
                fvg_low = float(right["high"])
                if fvg_low >= fvg_high:
                    continue
                midpoint = fvg_high - (fvg_high - fvg_low) * self.entry_fill_ratio
                price_in_gap = midpoint <= last_close <= fvg_high

            fvg_size = fvg_high - fvg_low
            if fvg_size < atr_value * self.min_fvg_atr_multiple:
                continue
            if not price_in_gap:
                continue

            # 확인 캔들 체크
            confirmed = (
                bullish_confirmation(df_3m.tail(2)) if side == Side.LONG
                else bearish_confirmation(df_3m.tail(2))
            )
            if not confirmed:
                continue

            return self._build_signal(
                context=context,
                df_3m=df_3m,
                df_15m=df_15m,
                atr_value=atr_value,
                side=side,
                fvg_low=fvg_low,
                fvg_high=fvg_high,
                fvg_midpoint=midpoint,
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
        fvg_low: float,
        fvg_high: float,
        fvg_midpoint: float,
    ) -> StrategySignal | None:
        last_close = float(df_3m["close"].iloc[-1])
        structure = build_structure_snapshot(df_entry=df_3m, df_structure=df_15m)

        setup_context: dict = {
            "side": side,
            "entry_price": last_close,
            "atr": atr_value,
            "breakout_level": fvg_high if side == Side.LONG else fvg_low,
            "range_boundaries": structure["range_boundaries"],
            "recent_swings_entry": structure["recent_swings_entry"],
            "recent_swings_structure": structure["recent_swings_structure"],
            "retest_low": fvg_low if side == Side.LONG else None,
            "retest_high": fvg_high if side == Side.SHORT else None,
            # 패턴 무효화: FVG 공백 경계
            "pattern_invalidation_low": fvg_low if side == Side.LONG else None,
            "pattern_invalidation_high": fvg_high if side == Side.SHORT else None,
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

        # 추세장에서 신뢰도 상향
        confidence = 0.72 if context.regime in {RegimeType.TRENDING, RegimeType.EXPANSION} else 0.62

        fvg_size = fvg_high - fvg_low
        entry_reason = build_entry_reason(
            title=f"FVG 공백 진입 ({'롱' if side == Side.LONG else '숏'})",
            lines=[
                f"FVG 구간 {fvg_low:,.4f} ~ {fvg_high:,.4f} (크기 {fvg_size:.4f})",
                f"가격이 FVG 중간선 {fvg_midpoint:,.4f} 에 진입 — 기관 미채움 구간.",
                f"확인 캔들 완성, 구조적 손절 {stop:,.4f} ({stop_reason})",
            ],
            chart_levels=[
                build_chart_level("FVG 상단", fvg_high, color="#f59e0b"),
                build_chart_level("FVG 하단", fvg_low, color="#f59e0b", linestyle=":"),
                build_chart_level("FVG 중간선", fvg_midpoint, color="#a78bfa"),
            ],
            chart_zones=[
                build_chart_zone("FVG 공백", fvg_low, fvg_high, color="#f59e0b", alpha=0.12),
            ],
            chart_marker=build_chart_marker(
                "FVG 진입",
                last_close,
                candle_time=df_3m.index[-1].isoformat(),
                color="#22c55e" if side == Side.LONG else "#ef4444",
            ),
        )
        # 지정가 진입: FVG 경계에서 대기
        if side == Side.SHORT:
            _optimal = fvg_high if fvg_high > last_close else None
        else:
            _optimal = fvg_low if fvg_low < last_close else None

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
            optimal_entry_price=_optimal,
            score=min(0.96, confidence + 0.06),
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
            tags=["fvg", "fair_value_gap", "smc", "institutional"],
            rationale={
                **entry_reason,
                "fvg_low": fvg_low,
                "fvg_high": fvg_high,
                "fvg_midpoint": fvg_midpoint,
                "fvg_size": fvg_size,
                "atr": atr_value,
                "stop_meta": stop_meta,
                "target_plan": targets,
                "regime": context.regime.value,
            },
        )
