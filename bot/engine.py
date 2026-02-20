from __future__ import annotations

import threading
from typing import Any, Callable, Optional


class TradingEngine:
    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self, target: Callable[..., Any], *args: Any, name: str = "BOT_ENGINE_THREAD", daemon: bool = True, **kwargs: Any) -> bool:
        if self._thread and self._thread.is_alive():
            return False
        self._running = True
        self._thread = threading.Thread(target=target, args=args, kwargs=kwargs, name=name, daemon=daemon)
        self._thread.start()
        return True

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def stop_flag(self) -> None:
        self._running = False

    @property
    def running(self) -> bool:
        return bool(self._running)
