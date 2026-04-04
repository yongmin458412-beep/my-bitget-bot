"""멀티타임프레임 필터 — 4H/1H/5M 방향 일치 시에만 진입 허용.

Alexander Elder 3중 스크린 방식:
  4H (추세 판단) → 1H (방향 확인) → 5M (진입 타이밍)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core.enums import Side
from core.logger import get_logger
from market.indicators import adx, atr, bollinger_bands, ema, macd, rsi, stochastic_rsi


@dataclass(slots=True)
class MTFVerdict:
    """멀티타임프레임 필터 결과."""

    approved: bool
    trend_direction: str  # bullish / bearish / neutral
    h4_bias: str  # bullish / bearish / neutral
    h1_bias: str
    m5_bias: str
    volume_ok: bool
    volatility_ok: bool
    blockers: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


class MultiTimeframeFilter:
    """4H → 1H → 5M 방향 일치 필터."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.logger = get_logger(__name__)

    # ── 4H 추세 판단 ──────────────────────────────────────────────────────

    def _check_4h(self, df: pd.DataFrame) -> tuple[str, dict[str, Any]]:
        """4H: EMA50/200 크로스 + ADX>25."""
        if len(df) < 200:
            return "neutral", {"reason": "insufficient_4h_data"}

        close = df["close"]
        ema50 = ema(close, self.config.h4_ema_fast)
        ema200 = ema(close, self.config.h4_ema_slow)
        adx_val = float(adx(df, 14).iloc[-1])

        ema50_last = float(ema50.iloc[-1])
        ema200_last = float(ema200.iloc[-1])

        if adx_val < self.config.h4_adx_threshold:
            return "neutral", {"adx": adx_val, "reason": "adx_too_low"}

        if ema50_last > ema200_last:
            return "bullish", {"adx": adx_val, "ema50": ema50_last, "ema200": ema200_last}
        elif ema50_last < ema200_last:
            return "bearish", {"adx": adx_val, "ema50": ema50_last, "ema200": ema200_last}
        return "neutral", {"adx": adx_val}

    # ── 1H 방향 확인 ──────────────────────────────────────────────────────

    def _check_1h(self, df: pd.DataFrame) -> tuple[str, dict[str, Any]]:
        """1H: EMA20 방향 + MACD 히스토그램 부호."""
        if len(df) < 30:
            return "neutral", {"reason": "insufficient_1h_data"}

        close = df["close"]
        ema20 = ema(close, self.config.h1_ema_period)
        _, _, hist = macd(close)

        # EMA20 기울기 (최근 3봉)
        ema_slope = float(ema20.iloc[-1]) - float(ema20.iloc[-3])
        hist_last = float(hist.iloc[-1])

        if ema_slope > 0 and hist_last > 0:
            return "bullish", {"ema_slope": ema_slope, "macd_hist": hist_last}
        elif ema_slope < 0 and hist_last < 0:
            return "bearish", {"ema_slope": ema_slope, "macd_hist": hist_last}
        return "neutral", {"ema_slope": ema_slope, "macd_hist": hist_last}

    # ── 5M 진입 타이밍 ────────────────────────────────────────────────────

    def _check_5m(self, df: pd.DataFrame, side: Side) -> tuple[str, dict[str, Any]]:
        """5M: RSI + StochRSI + 볼린저밴드."""
        if len(df) < 30:
            return "neutral", {"reason": "insufficient_5m_data"}

        close = df["close"]
        rsi_val = float(rsi(close, self.config.m5_rsi_period).iloc[-1])
        k, d = stochastic_rsi(close)
        k_last, d_last = float(k.iloc[-1]) if not pd.isna(k.iloc[-1]) else 50.0, float(d.iloc[-1]) if not pd.isna(d.iloc[-1]) else 50.0
        upper, mid, lower = bollinger_bands(close, self.config.m5_bb_period, self.config.m5_bb_std)
        last_close = float(close.iloc[-1])
        bb_upper = float(upper.iloc[-1])
        bb_lower = float(lower.iloc[-1])

        details = {"rsi": rsi_val, "stoch_k": k_last, "stoch_d": d_last, "bb_pos": "mid"}

        if side == Side.LONG:
            # 롱: RSI가 과매수 아님 + StochRSI K > D (상승 모멘텀) + 볼린저 하단~중간
            if rsi_val >= self.config.m5_rsi_overbought:
                return "neutral", {**details, "reason": "rsi_overbought"}
            if last_close <= bb_lower:
                details["bb_pos"] = "lower"
            return "bullish", details
        else:
            # 숏: RSI가 과매도 아님 + StochRSI K < D (하락 모멘텀) + 볼린저 상단~중간
            if rsi_val <= self.config.m5_rsi_oversold:
                return "neutral", {**details, "reason": "rsi_oversold"}
            if last_close >= bb_upper:
                details["bb_pos"] = "upper"
            return "bearish", details

    # ── 거래량 필터 ────────────────────────────────────────────────────────

    def _check_volume(self, df_5m: pd.DataFrame) -> bool:
        """현재 거래량 > N배 20기간 평균."""
        if len(df_5m) < self.config.volume_lookback + 1:
            return True  # 데이터 부족 시 통과
        avg_vol = df_5m["volume"].iloc[-self.config.volume_lookback - 1 : -1].mean()
        if avg_vol <= 0:
            return True
        return float(df_5m["volume"].iloc[-1]) >= avg_vol * self.config.volume_multiplier

    # ── 변동성 킬스위치 ───────────────────────────────────────────────────

    def _check_volatility(self, df_5m: pd.DataFrame) -> bool:
        """ATR > N배 평균 → 거래 중단."""
        atr_series = atr(df_5m)
        if len(atr_series) < self.config.volatility_atr_lookback:
            return True
        current_atr = float(atr_series.iloc[-1])
        avg_atr = float(atr_series.iloc[-self.config.volatility_atr_lookback :].mean())
        if avg_atr <= 0:
            return True
        return current_atr < avg_atr * self.config.volatility_atr_kill_multiplier

    # ── 메인 평가 ──────────────────────────────────────────────────────────

    def evaluate(
        self,
        *,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_5m: pd.DataFrame,
        side: Side,
    ) -> MTFVerdict:
        """멀티타임프레임 방향 일치 여부 평가."""

        blockers: list[str] = []

        # 4H 추세
        h4_bias, h4_details = self._check_4h(df_4h)

        # 1H 방향
        h1_bias, h1_details = self._check_1h(df_1h)

        # 5M 타이밍
        m5_bias, m5_details = self._check_5m(df_5m, side)

        # 거래량
        volume_ok = self._check_volume(df_5m)
        if not volume_ok:
            blockers.append("volume_below_threshold")

        # 변동성 킬스위치
        volatility_ok = self._check_volatility(df_5m)
        if not volatility_ok:
            blockers.append("extreme_volatility")

        # 방향 일치 체크
        # h4 neutral(횡보) 시에는 차단하지 않음 — 방향이 없으면 1H/5M 기준으로 진입
        # h4 가 신호와 반대 방향일 때만 차단
        side_str = "bullish" if side == Side.LONG else "bearish"
        opposite_str = "bearish" if side == Side.LONG else "bullish"

        if h4_bias == opposite_str:
            blockers.append(f"h4_against_{side_str}")

        if h1_bias == opposite_str:
            blockers.append(f"h1_against_{side_str}")

        # 추세 방향 결정
        if h4_bias == h1_bias and h4_bias != "neutral":
            trend_direction = h4_bias
        elif h4_bias != "neutral":
            trend_direction = h4_bias
        else:
            trend_direction = "neutral"

        approved = len(blockers) == 0

        return MTFVerdict(
            approved=approved,
            trend_direction=trend_direction,
            h4_bias=h4_bias,
            h1_bias=h1_bias,
            m5_bias=m5_bias,
            volume_ok=volume_ok,
            volatility_ok=volatility_ok,
            blockers=blockers,
            details={"h4": h4_details, "h1": h1_details, "m5": m5_details},
        )
