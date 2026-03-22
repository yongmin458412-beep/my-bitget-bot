"""Orderbook helper tests."""

from __future__ import annotations

from market.orderbook import DepthMetrics


def test_depth_metrics_to_payload_works_with_slots_dataclass() -> None:
    """Depth metrics should serialize without relying on __dict__."""

    metrics = DepthMetrics(symbol="BTCUSDT", spread_bps=1.2, top5_notional=150000.0, imbalance=0.15)
    payload = metrics.to_payload()
    assert payload["symbol"] == "BTCUSDT"
    assert payload["spread_bps"] == 1.2
    assert payload["top5_notional"] == 150000.0
    assert payload["imbalance"] == 0.15
