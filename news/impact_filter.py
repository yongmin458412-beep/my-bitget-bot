"""News-based entry blocking logic."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from core.persistence import SQLitePersistence


class NewsImpactFilter:
    """Query recent AI news analyses and decide whether entries should be blocked."""

    def __init__(self, persistence: SQLitePersistence) -> None:
        self.persistence = persistence

    def active_blocks(self, symbol: str) -> list[dict[str, str]]:
        """Return active news blocks for a symbol."""

        rows = self.persistence.fetchall(
            """
            SELECT n.title, n.published_at, a.summary_ko, a.impacted_assets_json, a.impact_level, a.direction_bias,
                   a.validity_window_minutes, a.should_block_new_entries, a.created_at
            FROM ai_news_analysis a
            JOIN news_items n ON n.news_hash = a.news_hash
            ORDER BY a.created_at DESC
            LIMIT 50
            """
        )
        now = datetime.now(tz=UTC)
        active: list[dict[str, str]] = []
        for row in rows:
            anchor_raw = str(row.get("published_at") or row.get("created_at") or "")
            try:
                anchor_time = datetime.fromisoformat(anchor_raw.replace("Z", "+00:00"))
            except ValueError:
                anchor_time = now
            validity = int(row.get("validity_window_minutes") or 0)
            if anchor_time + timedelta(minutes=validity) < now:
                continue
            if not bool(row.get("should_block_new_entries")):
                continue
            impacted_assets = json.loads(row.get("impacted_assets_json") or "[]")
            base_asset = symbol.upper().replace("USDT", "").replace("USDC", "").replace("USD", "")
            if (
                base_asset not in impacted_assets
                and symbol.upper() not in row["summary_ko"].upper()
                and symbol.upper() not in row["title"].upper()
            ):
                continue
            active.append(
                {
                    "title": row["title"],
                    "summary_ko": row["summary_ko"],
                    "impact_level": row["impact_level"],
                    "direction_bias": row["direction_bias"],
                }
            )
        return active
