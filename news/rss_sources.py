"""Default RSS and webpage sources for crypto and macro news."""

from __future__ import annotations

from dataclasses import dataclass

from core.enums import EventType


@dataclass(slots=True)
class SourceConfig:
    """News source definition."""

    name: str
    url: str
    kind: str
    event_type: EventType


CRYPTO_RSS_SOURCES = [
    SourceConfig("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/", "rss", EventType.NEWS),
    SourceConfig("Cointelegraph", "https://cointelegraph.com/rss", "rss", EventType.NEWS),
    SourceConfig("The Block", "https://www.theblock.co/rss.xml", "rss", EventType.NEWS),
]

EXCHANGE_NOTICE_SOURCES = [
    SourceConfig("Bitget Notice", "https://www.bitget.com/support/", "html", EventType.EXCHANGE_NOTICE),
]

ECONOMIC_FEEDS = [
    SourceConfig("Federal Reserve", "https://www.federalreserve.gov/feeds/press_all.xml", "rss", EventType.ECONOMIC),
    SourceConfig("BLS", "https://www.bls.gov/feed/bls_latest.rss", "rss", EventType.ECONOMIC),
]


def default_sources() -> list[SourceConfig]:
    """Return all default sources."""

    return [*CRYPTO_RSS_SOURCES, *EXCHANGE_NOTICE_SOURCES, *ECONOMIC_FEEDS]

