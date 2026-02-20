from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from diskcache import Cache
except Exception:
    Cache = None


@dataclass
class AICacheItem:
    value: Dict[str, Any]
    stored_epoch: float
    ttl_sec: int


class AICache:
    """
    Lightweight AI decision cache.
    - Uses diskcache when available (survives Streamlit reruns)
    - Falls back to in-memory dict when diskcache is unavailable
    """

    def __init__(self, cache_dir: str = "cache/ai", namespace: str = "trade") -> None:
        self._lock = threading.RLock()
        self._mem: Dict[str, AICacheItem] = {}
        self._disk = None
        if Cache is not None:
            try:
                self._disk = Cache(cache_dir)
            except Exception:
                self._disk = None
        self._namespace = str(namespace or "trade")

    def _k(self, key: str) -> str:
        return f"{self._namespace}:{str(key or '').strip()}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        kk = self._k(key)
        now = time.time()
        with self._lock:
            if self._disk is not None:
                try:
                    row = self._disk.get(kk, default=None)
                    if isinstance(row, dict):
                        return dict(row)
                except Exception:
                    pass
            item = self._mem.get(kk)
            if not isinstance(item, AICacheItem):
                return None
            if now > float(item.stored_epoch + max(1, int(item.ttl_sec))):
                self._mem.pop(kk, None)
                return None
            return dict(item.value)

    def set(self, key: str, value: Dict[str, Any], ttl_sec: int = 600) -> None:
        kk = self._k(key)
        ttl = int(max(1, ttl_sec))
        payload = dict(value or {})
        with self._lock:
            if self._disk is not None:
                try:
                    self._disk.set(kk, payload, expire=ttl)
                except Exception:
                    pass
            self._mem[kk] = AICacheItem(value=payload, stored_epoch=time.time(), ttl_sec=ttl)
            if len(self._mem) > 4000:
                # trim oldest
                keys = sorted(self._mem.items(), key=lambda kv: float(kv[1].stored_epoch))
                for old_k, _ in keys[:1000]:
                    self._mem.pop(old_k, None)

    def delete(self, key: str) -> None:
        kk = self._k(key)
        with self._lock:
            self._mem.pop(kk, None)
            if self._disk is not None:
                try:
                    self._disk.pop(kk, None)
                except Exception:
                    pass


def select_top_k_candidates(
    candidates: Iterable[Dict[str, Any]],
    *,
    k: int = 5,
    score_key: str = "local_score",
) -> List[Dict[str, Any]]:
    rows: List[Tuple[float, Dict[str, Any]]] = []
    for c in (candidates or []):
        if not isinstance(c, dict):
            continue
        try:
            score = float(c.get(score_key, 0.0) or 0.0)
        except Exception:
            score = 0.0
        rows.append((score, dict(c)))
    rows.sort(key=lambda x: float(x[0]), reverse=True)
    limit = int(max(1, k))
    return [dict(x[1]) for x in rows[:limit]]

