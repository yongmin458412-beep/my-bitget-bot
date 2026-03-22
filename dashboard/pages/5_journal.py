"""Journal page."""

from __future__ import annotations

import streamlit as st

from dashboard.common import run_query


st.title("Journal")
symbols = run_query("SELECT DISTINCT symbol FROM trades ORDER BY symbol")
selected_symbol = st.selectbox("심볼 필터", ["전체", *[row["symbol"] for row in symbols]])

if selected_symbol == "전체":
    query = "SELECT * FROM trades ORDER BY id DESC LIMIT 200"
    params = ()
else:
    query = "SELECT * FROM trades WHERE symbol = ? ORDER BY id DESC LIMIT 200"
    params = (selected_symbol,)

rows = run_query(query, params)
st.dataframe(rows, use_container_width=True)
