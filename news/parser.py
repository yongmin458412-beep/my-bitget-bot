"""News parsing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
import re

from core.enums import EventType
from core.utils import hash_text


ASSET_KEYWORDS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "SOL": ["solana", "sol"],
    "XRP": ["xrp", "ripple"],
    "DOGE": ["doge", "dogecoin"],
    "TOTAL": ["crypto market", "digital asset", "futures market"],
}


@dataclass(slots=True)
class ParsedNewsItem:
    """Normalized news/event item."""

    news_hash: str
    source: str
    title: str
    url: str
    published_at: str
    content: str
    related_assets: list[str]
    event_type: str


def parse_timestamp(value: str | None) -> str:
    """Normalize feed timestamps into ISO strings."""

    if not value:
        return datetime.now(tz=UTC).isoformat(timespec="seconds")
    try:
        return parsedate_to_datetime(value).astimezone(UTC).isoformat(timespec="seconds")
    except (TypeError, ValueError):
        return datetime.now(tz=UTC).isoformat(timespec="seconds")


def extract_related_assets(text: str) -> list[str]:
    """Extract related assets via keyword matching."""

    lowered = text.lower()
    assets: list[str] = []
    for asset, keywords in ASSET_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            assets.append(asset)
    return assets or ["BTC", "ETH"]


def normalize_entry(
    *,
    source: str,
    title: str,
    url: str,
    published_at: str | None,
    content: str,
    event_type: EventType,
) -> ParsedNewsItem:
    """Create a normalized news item."""

    merged = f"{title}\n{content}"
    related_assets = extract_related_assets(merged)
    return ParsedNewsItem(
        news_hash=hash_text(f"{source}|{title}|{url}|{parse_timestamp(published_at)}"),
        source=source,
        title=re.sub(r"\s+", " ", title).strip(),
        url=url,
        published_at=parse_timestamp(published_at),
        content=re.sub(r"\s+", " ", content).strip(),
        related_assets=related_assets,
        event_type=event_type.value,
    )

