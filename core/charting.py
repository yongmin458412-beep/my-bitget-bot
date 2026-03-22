"""Dark themed trade chart rendering for Telegram notifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from matplotlib import font_manager

from .utils import ensure_directory, hash_text


@dataclass(slots=True)
class TradeChartSpec:
    """Inputs required to render a trade chart image."""

    symbol: str
    product_type: str
    timeframe: str
    event_label: str
    mode: str
    side: str
    entry_price: float
    current_price: float
    stop_price: float | None = None
    tp1_price: float | None = None
    tp2_price: float | None = None
    tp3_price: float | None = None
    final_target_price: float | None = None
    quantity: float | None = None
    entry_notional_usdt: float | None = None
    remaining_notional_usdt: float | None = None
    leverage: float | None = None
    strategy_name: str = ""
    current_regime: str = ""
    conflict_resolution_summary: str = ""
    stop_reason: str = ""
    target_reasons: list[str] = field(default_factory=list)
    entry_reason_title: str = ""
    entry_reason_lines: list[str] = field(default_factory=list)
    chart_levels: list[dict[str, object]] = field(default_factory=list)
    chart_zones: list[dict[str, object]] = field(default_factory=list)
    chart_marker: dict[str, object] = field(default_factory=dict)
    rr_to_tp1: float | None = None
    rr_to_tp2: float | None = None
    indicators: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


_FONT_READY = False
_FONT_NAME = "DejaVu Sans"

_TERM_MAP = {
    "break_retest": "브레이크 앤 리테스트",
    "liquidity_raid": "유동성 스윕 반전",
    "manual_demo": "수동 데모",
    "session_breakout": "세션 레인지 돌파",
    "momentum_pullback": "모멘텀 눌림/플래그 재개",
    "trend": "추세장",
    "range": "횡보장",
    "expansion": "확장장",
    "event_risk": "이벤트 위험 구간",
    "dead_market": "죽은 장",
    "same_direction_dedupe": "동일 방향 중복 병합",
    "no_trade_opposite_conflict": "반대 신호 충돌로 진입 보류",
    "highest_score_after_opposite_conflict": "반대 신호 중 최고 점수 채택",
    "single_candidate": "단일 시그널 채택",
    "retest_low_atr_buffer": "리테스트 저점 이탈 + ATR 버퍼",
    "retest_high_atr_buffer": "리테스트 고점 돌파 + ATR 버퍼",
    "recent_swing_low": "최근 스윙 저점",
    "recent_swing_high": "최근 스윙 고점",
    "higher_timeframe_swing_high": "상위 타임프레임 스윙 고점",
    "higher_timeframe_swing_low": "상위 타임프레임 스윙 저점",
    "previous_high": "전고점",
    "previous_low": "전저점",
    "asia_high": "아시아 세션 고점",
    "asia_low": "아시아 세션 저점",
    "london_high": "런던 세션 고점",
    "london_low": "런던 세션 저점",
    "range_opposite_side": "레인지 반대편 경계",
    "equal_highs": "이퀄 하이",
    "equal_lows": "이퀄 로우",
    "partial_tp1": "1차 분할익절",
    "partial_tp2": "2차 분할익절",
    "partial_tp3": "3차 분할익절",
    "final_target_exit": "최종목표 도달 전량익절",
    "stop_out": "손절 청산",
    "time_stop": "시간 제한 청산",
    "tp3_runner": "러너 청산",
}


def _ensure_korean_font() -> str:
    """Pick a Korean-capable font when possible."""

    global _FONT_READY, _FONT_NAME
    if _FONT_READY:
        return _FONT_NAME

    _FONT_READY = True
    preferred = [
        "NanumGothic",
        "Noto Sans CJK KR",
        "NotoSansCJKkr",
        "Malgun Gothic",
        "AppleGothic",
        "Apple SD Gothic Neo",
    ]
    found_name = ""
    try:
        names = [str(getattr(item, "name", "") or "") for item in font_manager.fontManager.ttflist]
        name_map = {name.lower(): name for name in names if name}
        for candidate in preferred:
            if candidate.lower() in name_map:
                found_name = name_map[candidate.lower()]
                break
        if not found_name:
            hint_words = ["nanum", "notosanscjk", "malgun", "applegothic", "apple sd gothic", "noto sans kr", "noto sans cjk"]
            for path in (font_manager.findSystemFonts(fontpaths=None, fontext="ttf") or [])[:2000]:
                lower = Path(path).name.lower()
                if any(word in lower for word in hint_words):
                    try:
                        font_manager.fontManager.addfont(path)
                    except Exception:
                        continue
            names = [str(getattr(item, "name", "") or "") for item in font_manager.fontManager.ttflist]
            name_map = {name.lower(): name for name in names if name}
            for candidate in preferred:
                if candidate.lower() in name_map:
                    found_name = name_map[candidate.lower()]
                    break
    except Exception:
        found_name = ""

    _FONT_NAME = found_name or "DejaVu Sans"
    plt.rcParams["font.family"] = [_FONT_NAME]
    plt.rcParams["axes.unicode_minus"] = False
    return _FONT_NAME


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Return an EMA series."""

    return series.ewm(span=span, adjust=False).mean()


def _vwap(df: pd.DataFrame) -> pd.Series:
    """Compute session-like VWAP for display."""

    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_value = (typical * df["volume"]).cumsum()
    cum_volume = df["volume"].replace(0, np.nan).cumsum()
    return (cum_value / cum_volume).ffill().fillna(df["close"])


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ATR for annotation."""

    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _ko_term(value: str | None) -> str:
    """Translate internal reason/strategy tokens into Korean."""

    raw = str(value or "").strip()
    if not raw:
        return "-"
    if raw in _TERM_MAP:
        return _TERM_MAP[raw]
    return raw.replace("_", " ")


def _ko_terms(values: list[str]) -> list[str]:
    """Translate a list of internal tokens into Korean."""

    return [_ko_term(item) for item in values if str(item or "").strip()]


def _ko_note(value: str) -> str:
    """Translate any known internal tokens embedded in free-form note text."""

    rendered = str(value)
    for token, ko in _TERM_MAP.items():
        rendered = rendered.replace(token, ko)
    return rendered


_used_label_y: list[float] = []  # 레이블 y좌표 충돌 방지용 (render_trade_chart 호출마다 초기화)


def _annotate_level(
    ax: plt.Axes,
    value: float | None,
    color: str,
    label: str,
    *,
    linestyle: str = "--",
    percent_label: str | None = None,
    min_gap_frac: float = 0.012,  # Y축 전체 범위 대비 최소 간격 비율
) -> None:
    """Draw a level line and a collision-avoiding left-side label."""

    if value is None or not np.isfinite(value):
        return
    ax.axhline(value, color=color, linewidth=1.4, linestyle=linestyle, alpha=0.95, zorder=3)

    # 충돌 방지: 기존 레이블과 너무 가까우면 y를 밀어냄
    y_lo, y_hi = ax.get_ylim()
    y_range = max(y_hi - y_lo, 1e-9)
    min_gap = y_range * min_gap_frac
    label_y = value
    for used in sorted(_used_label_y, key=lambda v: abs(v - value)):
        if abs(label_y - used) < min_gap:
            direction = 1 if label_y >= used else -1
            label_y = used + direction * min_gap * 1.1
    _used_label_y.append(label_y)

    transform = ax.get_yaxis_transform()
    label_text = f"{label}  {value:,.4f}"
    if percent_label:
        label_text = f"{label}  {value:,.4f}  {percent_label}"
    ax.text(
        0.01,
        label_y,
        label_text,
        transform=transform,
        va="center",
        ha="left",
        fontsize=8.5,
        fontweight="bold",
        color=color,
        bbox={"boxstyle": "round,pad=0.20", "facecolor": "#0b1220", "edgecolor": color, "alpha": 0.85},
    )


def _draw_context_level(ax: plt.Axes, payload: dict[str, object]) -> None:
    """Draw an additional strategy-specific reference line."""

    try:
        price = float(payload.get("price"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return
    if not np.isfinite(price):
        return
    color = str(payload.get("color") or "#94a3b8")
    linestyle = str(payload.get("linestyle") or "--")
    label = str(payload.get("label") or "")
    ax.axhline(price, color=color, linewidth=1.0, linestyle=linestyle, alpha=0.75, zorder=2)
    if label:
        ax.text(
            0.005,
            price,
            label,
            transform=ax.get_yaxis_transform(),
            va="bottom",
            ha="left",
            fontsize=7.8,
            color=color,
            bbox={"boxstyle": "round,pad=0.14", "facecolor": "#0b1220", "edgecolor": color, "alpha": 0.78},
        )


def _draw_context_zone(ax: plt.Axes, payload: dict[str, object]) -> None:
    """Draw a horizontal strategy-specific zone."""

    try:
        low = float(payload.get("low"))  # type: ignore[arg-type]
        high = float(payload.get("high"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return
    if not np.isfinite(low) or not np.isfinite(high):
        return
    label = str(payload.get("label") or "")
    color = str(payload.get("color") or "#38bdf8")
    alpha = float(payload.get("alpha") or 0.10)
    lower = min(low, high)
    upper = max(low, high)
    ax.axhspan(lower, upper, color=color, alpha=alpha, zorder=1)
    if label:
        ax.text(
            0.16,
            upper,
            label,
            transform=ax.get_yaxis_transform(),
            va="bottom",
            ha="left",
            fontsize=7.3,
            color=color,
        )


def _draw_context_marker(ax: plt.Axes, frame: pd.DataFrame, payload: dict[str, object]) -> None:
    """Draw a strategy-specific confirmation marker near the triggering candle."""

    if not payload:
        return
    try:
        price = float(payload.get("price"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return
    if not np.isfinite(price):
        return
    label = str(payload.get("label") or "")
    color = str(payload.get("color") or "#f8fafc")
    marker = str(payload.get("marker") or "o")
    candle_time = payload.get("candle_time")
    if candle_time:
        try:
            timestamp = pd.Timestamp(candle_time).tz_localize(None) if pd.Timestamp(candle_time).tzinfo else pd.Timestamp(candle_time)
        except Exception:
            timestamp = frame.index[-1]
    else:
        timestamp = frame.index[-1]
    x_value = _resolve_marker_index(frame.index, timestamp)
    ax.scatter([x_value], [price], s=62, marker=marker, color=color, edgecolors="#ffffff", linewidths=0.9, zorder=4)
    if label:
        ax.annotate(
            label,
            xy=(x_value, price),
            xytext=(10, 12),
            textcoords="offset points",
            fontsize=7.8,
            color=color,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "#0b1220", "edgecolor": color, "alpha": 0.78},
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8},
        )


def _resolve_marker_index(index: pd.DatetimeIndex, timestamp: pd.Timestamp) -> int:
    """Map a candle timestamp onto mplfinance's positional x-axis."""

    if index.empty:
        return 0
    target = pd.Timestamp(timestamp)
    if target.tzinfo is not None:
        target = target.tz_localize(None)
    differences = np.abs((index - target).asi8)
    nearest = int(np.argmin(differences))
    return max(0, min(nearest, len(index) - 1))


def _nearest_levels(frame: pd.DataFrame, current_price: float, *, count: int = 3) -> tuple[list[float], list[float]]:
    """Approximate nearby support and resistance levels."""

    lows = sorted({round(float(value), 6) for value in frame["low"].tail(90).tolist() if np.isfinite(value)})
    highs = sorted({round(float(value), 6) for value in frame["high"].tail(90).tolist() if np.isfinite(value)})
    supports = [value for value in lows if value <= current_price][-count:]
    resistances = [value for value in highs if value >= current_price][:count]
    return supports, resistances


def _volume_profile_nodes(frame: pd.DataFrame, *, bins: int = 32, top_n: int = 3) -> list[float]:
    """Return rough high-volume price nodes using close/volume bins."""

    closes = frame["close"].to_numpy(dtype=float)
    volumes = frame["volume"].to_numpy(dtype=float)
    if len(closes) < 10:
        return []
    low = float(np.nanmin(closes))
    high = float(np.nanmax(closes))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return []
    edges = np.linspace(low, high, bins + 1)
    hist = np.zeros(bins, dtype=float)
    indices = np.digitize(closes, edges) - 1
    indices = np.clip(indices, 0, bins - 1)
    for idx, volume in zip(indices, volumes, strict=False):
        hist[idx] += float(max(volume, 0.0))
    top_indices = np.argsort(hist)[-top_n:]
    centers = []
    for idx in sorted(top_indices):
        center = float((edges[idx] + edges[idx + 1]) / 2.0)
        if np.isfinite(center):
            centers.append(center)
    return centers


def _summary_text(spec: TradeChartSpec) -> str:
    """Build the compact top-right info badge (strategy | leverage | RR)."""

    parts: list[str] = []
    if spec.strategy_name:
        parts.append(_ko_term(spec.strategy_name))
    if spec.leverage is not None:
        parts.append(f"레버 {spec.leverage:.0f}배")
    rr_parts: list[str] = []
    if spec.rr_to_tp1 is not None:
        rr_parts.append(f"1R:{spec.rr_to_tp1:.2f}")
    if spec.rr_to_tp2 is not None:
        rr_parts.append(f"2R:{spec.rr_to_tp2:.2f}")
    if rr_parts:
        parts.append(" / ".join(rr_parts))
    return "  |  ".join(parts)


def render_trade_chart(
    df: pd.DataFrame,
    spec: TradeChartSpec,
    *,
    output_dir: str | Path,
) -> Path:
    """Render a PNG snapshot for Telegram delivery."""

    if df.empty:
        raise ValueError("차트 렌더링용 캔들 데이터가 비어 있습니다.")

    _ensure_korean_font()
    output_root = ensure_directory(output_dir)
    frame = df.copy().tail(140).sort_index()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None)
    frame["ema20"] = _ema(frame["close"], 20)
    frame["ema50"] = _ema(frame["close"], 50)
    frame["atr14"] = _atr(frame, 14)

    mpf_frame = pd.DataFrame(
        {
            "Open": frame["open"].astype(float),
            "High": frame["high"].astype(float),
            "Low": frame["low"].astype(float),
            "Close": frame["close"].astype(float),
            "Volume": frame["volume"].astype(float),
        },
        index=frame.index,
    )

    market_colors = mpf.make_marketcolors(
        up="#00d084",
        down="#ff4d4f",
        edge={"up": "#00d084", "down": "#ff4d4f"},
        wick={"up": "#00d084", "down": "#ff4d4f"},
        volume={"up": "#00d084", "down": "#ff4d4f"},
        ohlc={"up": "#00d084", "down": "#ff4d4f"},
    )
    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        marketcolors=market_colors,
        facecolor="#11161c",
        figcolor="#0e1117",
        edgecolor="#263238",
        gridcolor="#2d3238",
        gridstyle="--",
        rc={
            "font.family": _FONT_NAME,
            "axes.labelcolor": "#94a3b8",
            "xtick.color": "#cfd8dc",
            "ytick.color": "#cfd8dc",
            "text.color": "#e5e7eb",
            "axes.titlecolor": "#ffffff",
        },
    )
    addplots = [
        mpf.make_addplot(frame["ema20"].to_numpy(dtype=float), color="#f59e0b", width=1.0, alpha=0.6, panel=0),
        mpf.make_addplot(frame["ema50"].to_numpy(dtype=float), color="#60a5fa", width=0.9, alpha=0.6, panel=0),
    ]
    fig, axes = mpf.plot(
        mpf_frame,
        type="candle",
        style=style,
        volume=True,
        addplot=addplots,
        panel_ratios=(7.5, 1.35),
        figratio=(13, 8),
        figscale=1.0,
        returnfig=True,
        datetime_format="%m-%d %H:%M",
        xrotation=0,
        tight_layout=False,
    )
    fig.set_dpi(130)
    ax_price = axes[0]
    ax_volume = axes[2]
    for axis in (ax_price, ax_volume):
        axis.grid(True, color="#2d3238", linestyle="--", linewidth=0.45, alpha=0.32)
        axis.tick_params(colors="#cfd8dc", labelsize=8)
        for spine in axis.spines.values():
            spine.set_color("#263238")
            spine.set_linewidth(0.8)

    if spec.chart_marker:
        _draw_context_marker(ax_price, frame, spec.chart_marker)

    # ── Y축 줌: 진입/TP/SL 구간 중심으로 확대 ─────────────────────────────
    key_prices = [p for p in [
        spec.entry_price, spec.stop_price,
        spec.tp1_price, spec.tp2_price, spec.tp3_price,
    ] if p is not None and np.isfinite(p)]
    if key_prices:
        atr_val = float(frame["atr14"].iloc[-1]) if "atr14" in frame.columns and not frame["atr14"].empty else (
            abs(max(key_prices) - min(key_prices)) * 0.5 or spec.entry_price * 0.005
        )
        padding = max(atr_val * 4, abs(max(key_prices) - min(key_prices)) * 0.4)
        y_lo = min(key_prices) - padding
        y_hi = max(key_prices) + padding
        ax_price.set_ylim(y_lo, y_hi)

    # ── 컨텍스트 레벨/존: set_ylim 이후에 그려야 범위 밖 레이블이 표시되지 않음 ──
    _ylim_lo, _ylim_hi = ax_price.get_ylim()
    for zone in spec.chart_zones:
        if isinstance(zone, dict):
            _draw_context_zone(ax_price, zone)
    for level in spec.chart_levels:
        if isinstance(level, dict):
            try:
                _lv_price = float(level.get("price"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                _lv_price = None
            if _lv_price is not None and (_lv_price < _ylim_lo or _lv_price > _ylim_hi):
                continue  # Y축 범위 밖 레이블 생략
            _draw_context_level(ax_price, level)

    # Compute actual % move labels relative to entry
    def _pct(target: float | None) -> str | None:
        if target is None or not np.isfinite(target) or spec.entry_price == 0:
            return None
        pct = (target - spec.entry_price) / spec.entry_price * 100
        return f"{pct:+.2f}%"

    # 레이블 충돌 방지용 y좌표 목록 초기화 (렌더링마다 새로 시작)
    _used_label_y.clear()
    _annotate_level(ax_price, spec.entry_price, "#00bcd4", "진입", linestyle="-")
    _annotate_level(ax_price, spec.stop_price, "#ff4d4f", "손절", percent_label=_pct(spec.stop_price))
    _annotate_level(ax_price, spec.tp1_price, "#2dd4bf", "TP1", linestyle="-.", percent_label=_pct(spec.tp1_price))
    _annotate_level(ax_price, spec.tp2_price, "#00d084", "TP2", percent_label=_pct(spec.tp2_price))
    _annotate_level(ax_price, spec.tp3_price, "#22c55e", "TP3", percent_label=_pct(spec.tp3_price))

    ax_price.set_ylabel("")
    ax_volume.set_ylabel("")

    summary_text = _summary_text(spec)
    if summary_text:
        ax_price.text(
            0.5,
            0.985,
            summary_text,
            transform=ax_price.transAxes,
            va="top",
            ha="center",
            fontsize=9.0,
            fontweight="bold",
            color="#e5e7eb",
            bbox={"boxstyle": "round,pad=0.30", "facecolor": "#0b1220", "edgecolor": "#475569", "alpha": 0.88},
        )

    side_text = "▲ LONG" if spec.side == "long" else "▼ SHORT"
    side_color = "#00d084" if spec.side == "long" else "#ff4d4f"
    title = f"{spec.symbol}  {side_text}  [{spec.timeframe}]  {spec.event_label}"
    ax_price.set_title(title, color=side_color, fontsize=12, fontweight="bold", pad=8)

    fig.subplots_adjust(left=0.04, right=0.92, top=0.93, bottom=0.06, hspace=0.03)

    filename = f"{spec.symbol}_{hash_text(f'{spec.event_label}|{spec.entry_price}|{datetime.now(tz=UTC).isoformat()}')[:10]}.png"
    path = output_root / filename
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
