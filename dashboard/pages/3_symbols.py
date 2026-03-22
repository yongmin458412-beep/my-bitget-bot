"""Symbols and active-universe page."""

from __future__ import annotations

import streamlit as st

from dashboard.common import render_candlestick, run_query


st.title("Symbols")
rows = run_query(
    """
    SELECT symbol, product_type, total_score, liquidity_score, volatility_score, spread_score, depth_score, trend_score
    FROM symbol_scores
    ORDER BY id DESC
    LIMIT 200
    """
)
st.dataframe(rows, use_container_width=True)

symbol = st.text_input("차트 심볼", value="BTCUSDT")
product_type = st.selectbox("상품군", ["USDT-FUTURES", "USDC-FUTURES", "COIN-FUTURES"])
if symbol:
    figure = render_candlestick(symbol, product_type=product_type)
    st.plotly_chart(figure, use_container_width=True)
