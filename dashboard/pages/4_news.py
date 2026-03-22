"""News page."""

from __future__ import annotations

import streamlit as st

from dashboard.common import run_query


st.title("News")
rows = run_query(
    """
    SELECT n.title, n.source, n.published_at, a.summary_ko, a.impact_level, a.direction_bias, a.should_block_new_entries
    FROM news_items n
    LEFT JOIN ai_news_analysis a ON a.news_hash = n.news_hash
    ORDER BY n.id DESC
    LIMIT 100
    """
)
st.dataframe(rows, use_container_width=True)
