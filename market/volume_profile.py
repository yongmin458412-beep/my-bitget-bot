"""Lightweight volume profile helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class VolumeProfile:
    """Simple histogram-based volume profile."""

    high_volume_nodes: list[float]
    low_volume_nodes: list[float]


def build_volume_profile(df: pd.DataFrame, bins: int = 24) -> VolumeProfile:
    """Approximate HVN/LVN levels using close-price histogram buckets."""

    if df.empty:
        return VolumeProfile(high_volume_nodes=[], low_volume_nodes=[])

    closes = df["close"].to_numpy()
    volumes = df["volume"].to_numpy()
    hist, edges = np.histogram(closes, bins=bins, weights=volumes)
    centers = (edges[:-1] + edges[1:]) / 2
    ranked = sorted(zip(hist, centers), key=lambda item: item[0], reverse=True)
    high_nodes = [float(center) for _, center in ranked[:3]]
    low_nodes = [float(center) for _, center in sorted(ranked[-3:], key=lambda item: item[0])]
    return VolumeProfile(high_volume_nodes=high_nodes, low_volume_nodes=low_nodes)

