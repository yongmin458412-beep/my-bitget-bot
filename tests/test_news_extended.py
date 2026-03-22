"""Extended news module tests: parser, dedup, impact filter."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from core.enums import EventType
from core.persistence import SQLitePersistence
from news.impact_filter import NewsImpactFilter
from news.parser import extract_related_assets, normalize_entry, parse_timestamp


# ── Parser tests ────────────────────────────────────────────────────────────

def test_extract_related_assets_btc_keyword() -> None:
    assert "BTC" in extract_related_assets("bitcoin price surges to 100k")


def test_extract_related_assets_eth_keyword() -> None:
    assert "ETH" in extract_related_assets("Ethereum upgrade expected this week")


def test_extract_related_assets_multiple() -> None:
    assets = extract_related_assets("BTC and ETH lead the market rally with solana")
    assert "BTC" in assets
    assert "ETH" in assets
    assert "SOL" in assets


def test_extract_related_assets_fallback_to_btc_eth() -> None:
    """Unknown text falls back to ['BTC', 'ETH']."""
    assets = extract_related_assets("general market commentary with no specific tokens")
    assert assets == ["BTC", "ETH"]


def test_normalize_entry_strips_whitespace() -> None:
    item = normalize_entry(
        source="test",
        title="  Bitcoin   Rally  ",
        url="https://example.com",
        published_at=None,
        content="Price   action",
        event_type=EventType.NEWS,
    )
    assert item.title == "Bitcoin Rally"
    assert item.content == "Price action"


def test_normalize_entry_hash_deterministic() -> None:
    """Same inputs should produce same hash."""
    published = "Thu, 20 Mar 2026 10:00:00 +0000"
    item1 = normalize_entry(
        source="CoinDesk", title="BTC hits 100k", url="https://cd.com/1",
        published_at=published, content="amazing", event_type=EventType.NEWS,
    )
    item2 = normalize_entry(
        source="CoinDesk", title="BTC hits 100k", url="https://cd.com/1",
        published_at=published, content="amazing", event_type=EventType.NEWS,
    )
    assert item1.news_hash == item2.news_hash


def test_normalize_entry_different_urls_different_hash() -> None:
    item1 = normalize_entry(
        source="CoinDesk", title="BTC", url="https://cd.com/1",
        published_at=None, content="", event_type=EventType.NEWS,
    )
    item2 = normalize_entry(
        source="CoinDesk", title="BTC", url="https://cd.com/2",
        published_at=None, content="", event_type=EventType.NEWS,
    )
    assert item1.news_hash != item2.news_hash


def test_parse_timestamp_rfc2822() -> None:
    ts = parse_timestamp("Thu, 20 Mar 2026 10:00:00 +0000")
    assert ts.startswith("2026-03-20")


def test_parse_timestamp_none_returns_now() -> None:
    ts = parse_timestamp(None)
    now = datetime.now(tz=UTC)
    parsed = datetime.fromisoformat(ts)
    assert abs((now - parsed).total_seconds()) < 5


def test_parse_timestamp_invalid_falls_back_to_now() -> None:
    ts = parse_timestamp("not-a-date")
    now = datetime.now(tz=UTC)
    parsed = datetime.fromisoformat(ts)
    assert abs((now - parsed).total_seconds()) < 5


# ── NewsCollector deduplication tests ───────────────────────────────────────

def test_filter_new_removes_known_hashes(tmp_path: Path) -> None:
    """Items already in DB should be filtered out."""

    from news.collector import NewsCollector

    persistence = SQLitePersistence(tmp_path / "news.sqlite3")
    collector = NewsCollector(persistence)

    # Insert one item directly
    now = datetime.now(tz=UTC).isoformat(timespec="seconds")
    persistence.execute(
        "INSERT INTO news_items (news_hash, source, title, url, published_at, content, related_assets_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("existing-hash", "test", "Old news", "https://ex.com", now, "", "[]", now),
    )

    from news.parser import ParsedNewsItem
    items = [
        ParsedNewsItem(
            news_hash="existing-hash",
            source="test", title="Old news", url="https://ex.com",
            published_at=now, content="", related_assets=[], event_type="price_action",
        ),
        ParsedNewsItem(
            news_hash="new-hash",
            source="test", title="New news", url="https://ex.com/2",
            published_at=now, content="", related_assets=[], event_type="price_action",
        ),
    ]

    fresh = collector._filter_new(items)
    assert len(fresh) == 1
    assert fresh[0].news_hash == "new-hash"


# ── NewsImpactFilter tests ───────────────────────────────────────────────────

def _insert_analysis(persistence: SQLitePersistence, *, symbol: str, block: bool, validity_minutes: int) -> None:
    """Helper to populate news_items + ai_news_analysis for testing."""

    now = datetime.now(tz=UTC).isoformat(timespec="seconds")
    news_hash = f"hash-{symbol}-{block}"
    persistence.execute(
        "INSERT OR IGNORE INTO news_items (news_hash, source, title, url, published_at, content, related_assets_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (news_hash, "test", f"{symbol} news", "https://ex.com", now, "", f'["{symbol}"]', now),
    )
    persistence.execute(
        "INSERT OR IGNORE INTO ai_news_analysis "
        "(news_hash, summary_ko, impacted_assets_json, impact_level, direction_bias, "
        "validity_window_minutes, confidence, event_type, should_block_new_entries, notes, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (news_hash, f"{symbol} test", f'["{symbol}"]', "high", "neutral",
         validity_minutes, 0.9, "economic", 1 if block else 0, "", now),
    )


def test_impact_filter_blocks_matching_symbol(tmp_path: Path) -> None:
    persistence = SQLitePersistence(tmp_path / "n.sqlite3")
    _insert_analysis(persistence, symbol="BTC", block=True, validity_minutes=120)

    filt = NewsImpactFilter(persistence)
    blocks = filt.active_blocks("BTCUSDT")
    assert len(blocks) >= 1
    assert blocks[0]["impact_level"] == "high"


def test_impact_filter_does_not_block_other_symbol(tmp_path: Path) -> None:
    persistence = SQLitePersistence(tmp_path / "n.sqlite3")
    _insert_analysis(persistence, symbol="ETH", block=True, validity_minutes=120)

    filt = NewsImpactFilter(persistence)
    blocks = filt.active_blocks("SOLUSDT")
    assert len(blocks) == 0


def test_impact_filter_ignores_expired_analyses(tmp_path: Path) -> None:
    """Items past their validity_window should not appear."""

    persistence = SQLitePersistence(tmp_path / "n.sqlite3")
    past = (datetime.now(tz=UTC) - timedelta(hours=5)).isoformat(timespec="seconds")
    news_hash = "expired-hash"
    persistence.execute(
        "INSERT OR IGNORE INTO news_items (news_hash, source, title, url, published_at, content, related_assets_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (news_hash, "test", "BTC old news", "https://ex.com", past, "", '["BTC"]', past),
    )
    persistence.execute(
        "INSERT OR IGNORE INTO ai_news_analysis "
        "(news_hash, summary_ko, impacted_assets_json, impact_level, direction_bias, "
        "validity_window_minutes, confidence, event_type, should_block_new_entries, notes, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (news_hash, "BTC test", '["BTC"]', "high", "neutral", 60, 0.9, "economic", 1, "", past),
    )

    filt = NewsImpactFilter(persistence)
    blocks = filt.active_blocks("BTCUSDT")
    assert len(blocks) == 0


def test_impact_filter_no_block_flag_passes(tmp_path: Path) -> None:
    persistence = SQLitePersistence(tmp_path / "n.sqlite3")
    _insert_analysis(persistence, symbol="BTC", block=False, validity_minutes=120)

    filt = NewsImpactFilter(persistence)
    blocks = filt.active_blocks("BTCUSDT")
    assert len(blocks) == 0
