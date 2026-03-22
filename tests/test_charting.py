"""Tests for Telegram trade chart rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.charting import TradeChartSpec, _resolve_marker_index, render_trade_chart


def test_render_trade_chart_creates_png(tmp_path: Path) -> None:
    """The renderer should save a non-empty PNG file."""

    index = pd.date_range("2026-03-18 00:00:00", periods=80, freq="5min", tz="UTC")
    base = np.linspace(100.0, 112.0, len(index))
    frame = pd.DataFrame(
        {
            "open": base,
            "high": base + 1.2,
            "low": base - 1.1,
            "close": base + np.sin(np.linspace(0, 6, len(index))),
            "volume": np.linspace(1000, 2400, len(index)),
            "quote_volume": np.linspace(120000, 300000, len(index)),
        },
        index=index,
    )
    spec = TradeChartSpec(
        symbol="BTCUSDT",
        product_type="USDT-FUTURES",
        timeframe="5m",
        event_label="진입",
        mode="DEMO",
        side="long",
        entry_price=110.0,
        current_price=111.2,
        stop_price=108.4,
        tp1_price=111.6,
        tp2_price=113.2,
        tp3_price=115.0,
        final_target_price=116.4,
        quantity=0.01,
        entry_notional_usdt=1.10,
        remaining_notional_usdt=1.11,
        leverage=25.0,
        strategy_name="break_retest",
        current_regime="trend",
        stop_reason="retest_low_atr_buffer",
        target_reasons=["previous_high", "session_high", "range_opposite_side"],
        entry_reason_title="전고점 돌파 후 리테스트 매수",
        entry_reason_lines=[
            "직전 전고점을 강하게 돌파했습니다.",
            "돌파 레벨을 다시 테스트한 뒤 지지 확인이 나왔습니다.",
            "확인 캔들이 붙어 롱 진입했습니다.",
        ],
        chart_levels=[
            {"label": "돌파 레벨", "price": 109.4, "color": "#f59e0b", "linestyle": "--"},
        ],
        chart_zones=[
            {"label": "리테스트 존", "low": 109.0, "high": 109.6, "color": "#38bdf8", "alpha": 0.10},
        ],
        chart_marker={
            "label": "확인 캔들",
            "price": 110.2,
            "candle_time": index[-5].isoformat(),
            "color": "#f8fafc",
            "marker": "o",
        },
        rr_to_tp1=1.0,
        rr_to_tp2=2.0,
        indicators=["EMA20", "EMA50", "VWAP", "RSI14", "ADX14", "ATR14"],
        notes=["전략: break_retest", "태그: break, retest"],
    )

    chart_path = render_trade_chart(frame, spec, output_dir=tmp_path)

    assert chart_path.exists()
    assert chart_path.suffix == ".png"
    assert chart_path.stat().st_size > 0


def test_resolve_marker_index_returns_positional_index() -> None:
    """Marker placement should map timestamps to mplfinance positional coordinates."""

    index = pd.date_range("2026-03-18 00:00:00", periods=10, freq="5min", tz="UTC").tz_localize(None)
    marker_idx = _resolve_marker_index(index, pd.Timestamp("2026-03-18 00:21:00"))
    assert isinstance(marker_idx, int)
    assert marker_idx == 4
