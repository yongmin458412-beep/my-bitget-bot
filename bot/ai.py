from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional


def normalize_budget_limit(v: Any) -> int:
    try:
        n = int(v)
    except Exception:
        n = 0
    return int(n if n > 0 else 0)


def ai_budget_window_note(
    *,
    allow: bool,
    reason_code: str,
    reason_note: str = "",
    next_allowed_sec: int = 0,
    next_allowed_kst: str = "",
) -> str:
    if allow:
        return str(reason_note or "ok")
    left = int(max(0, int(next_allowed_sec or 0)))
    kst = str(next_allowed_kst or "").strip()
    if kst:
        return f"{reason_code}: {reason_note} | 재개 {kst} ({left}s)"
    return f"{reason_code}: {reason_note} ({left}s)"


class AICache:
    def __init__(
        self,
        *,
        ttl_sec: int = 600,
        namespace: str = "default",
        cache_dir: str = ".cache/ai_cache",
    ) -> None:
        self._ttl_sec = int(max(10, min(86400, int(ttl_sec or 600))))
        self._namespace = str(namespace or "default").strip() or "default"
        self._lock = threading.RLock()
        self._mem: Dict[str, Dict[str, Any]] = {}
        self._disk = None
        try:
            from diskcache import Cache  # type: ignore

            self._disk = Cache(cache_dir)
        except Exception:
            self._disk = None

    def _full_key(self, key: str) -> str:
        return f"{self._namespace}:{str(key or '').strip()}"

    def set(self, key: str, value: Dict[str, Any], ttl_sec: Optional[int] = None) -> None:
        fk = self._full_key(key)
        ttl = int(max(10, min(86400, int(ttl_sec or self._ttl_sec))))
        payload = {
            "ts": float(time.time()),
            "ttl": int(ttl),
            "value": dict(value or {}),
        }
        with self._lock:
            self._mem[fk] = payload
            if self._disk is not None:
                try:
                    self._disk.set(fk, payload, expire=ttl)
                except Exception:
                    pass

    def get(self, key: str, max_age_sec: Optional[int] = None) -> Optional[Dict[str, Any]]:
        fk = self._full_key(key)
        now = float(time.time())
        max_age = int(max(10, min(86400, int(max_age_sec or self._ttl_sec))))
        with self._lock:
            payload = self._mem.get(fk)
            if payload is None and self._disk is not None:
                try:
                    payload = self._disk.get(fk, default=None)
                except Exception:
                    payload = None
            if not isinstance(payload, dict):
                return None
            ts = float(payload.get("ts", 0.0) or 0.0)
            ttl = int(payload.get("ttl", self._ttl_sec) or self._ttl_sec)
            age = now - ts if ts > 0 else 999999.0
            if age > float(min(ttl, max_age)):
                return None
            val = payload.get("value")
            return dict(val) if isinstance(val, dict) else None

    def purge(self) -> None:
        with self._lock:
            self._mem.clear()
            if self._disk is not None:
                try:
                    self._disk.clear()
                except Exception:
                    pass
