from __future__ import annotations

import math
import re
from typing import Any

try:
    import numpy as np
except Exception:
    np = None


def clamp(v: Any, lo: Any, hi: Any):
    try:
        return max(lo, min(hi, v))
    except Exception:
        return lo


def as_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        if isinstance(v, bool):
            return float(int(v))
        if np is not None and isinstance(v, (int, float, np.integer, np.floating)):
            x = float(v)
            if math.isnan(x) or math.isinf(x):
                return float(default)
            return x
        if isinstance(v, (int, float)):
            x = float(v)
            if math.isnan(x) or math.isinf(x):
                return float(default)
            return x
        s = str(v).strip()
        if not s:
            return float(default)
        if s.lower() in ["none", "null", "nan"]:
            return float(default)
        return float(s)
    except Exception:
        return float(default)


def as_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return int(default)
        if isinstance(v, bool):
            return int(v)
        if np is not None and isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, int):
            return int(v)
        return int(round(as_float(v, float(default))))
    except Exception:
        return int(default)


def timeframe_seconds(tf: str, default_sec: int = 300) -> int:
    try:
        s = str(tf or "").strip().lower()
        m = re.match(r"^(\d+)\s*([mhdw])$", s)
        if not m:
            return int(default_sec)
        n = int(m.group(1))
        u = m.group(2)
        if n <= 0:
            return int(default_sec)
        if u == "m":
            return int(n * 60)
        if u == "h":
            return int(n * 60 * 60)
        if u == "d":
            return int(n * 24 * 60 * 60)
        if u == "w":
            return int(n * 7 * 24 * 60 * 60)
        return int(default_sec)
    except Exception:
        return int(default_sec)
