"""Live trading adapter with explicit safety checks."""

from __future__ import annotations

from core.settings import AppSettings

from .bitget_demo import BitgetExchangeBase


class BitgetLiveExchange(BitgetExchangeBase):
    """Live trading exchange wrapper."""

    def __init__(self, settings: AppSettings) -> None:
        settings.ensure_live_trading_allowed()
        super().__init__(settings, demo=False)
