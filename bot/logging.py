from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional

EventSink = Callable[[Dict[str, Any]], None]

_SINK_LOCK = threading.RLock()
_SINKS: Dict[str, EventSink] = {}


def register_sink(name: str, sink: EventSink, overwrite: bool = True) -> bool:
    sink_name = str(name or "").strip()
    if not sink_name or not callable(sink):
        return False
    with _SINK_LOCK:
        if (not overwrite) and sink_name in _SINKS:
            return False
        _SINKS[sink_name] = sink
    return True


def unregister_sink(name: str) -> bool:
    sink_name = str(name or "").strip()
    if not sink_name:
        return False
    with _SINK_LOCK:
        if sink_name in _SINKS:
            _SINKS.pop(sink_name, None)
            return True
    return False


def list_sinks() -> List[str]:
    with _SINK_LOCK:
        return sorted(_SINKS.keys())


def emit_event(event_type: str, payload: Optional[Dict[str, Any]] = None, **meta: Any) -> Dict[str, Any]:
    event = {
        "event_type": str(event_type or "").strip().upper(),
        "payload": dict(payload or {}),
        "meta": dict(meta or {}),
        "ts_epoch": float(time.time()),
    }
    deliveries: List[Dict[str, Any]] = []
    with _SINK_LOCK:
        sinks = dict(_SINKS)
    for sink_name, sink_fn in sinks.items():
        try:
            sink_fn(dict(event))
            deliveries.append({"sink": sink_name, "ok": True})
        except Exception as e:
            deliveries.append({"sink": sink_name, "ok": False, "reason_code": type(e).__name__, "detail": str(e)[:200]})
    return {
        "event_type": event["event_type"],
        "sink_count": len(sinks),
        "deliveries": deliveries,
    }
