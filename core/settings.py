"""Configuration loading via defaults.json, bot_settings.json, and environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .enums import ProductType, TradingMode
from .time_utils import utc_now
from .utils import deep_merge, dump_json, load_json


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULTS_PATH = ROOT_DIR / "config" / "defaults.json"
BOT_SETTINGS_PATH = ROOT_DIR / "config" / "bot_settings.json"
ENV_PATH = ROOT_DIR / ".env"


class TimeframeConfig(BaseModel):
    """Timeframe settings used across the stack."""

    entry: str = "3m"
    confirm: str = "5m"
    structure: str = "15m"
    higher: str = "1H"     # 멀티타임프레임: 방향 확인
    macro: str = "4H"      # 멀티타임프레임: 추세 판단


class MTFConfig(BaseModel):
    """멀티타임프레임 필터 설정 (Alexander Elder 3중 스크린)."""

    enabled: bool = True
    # 4H 추세 판단
    h4_ema_fast: int = 50
    h4_ema_slow: int = 200
    h4_adx_threshold: float = 25.0
    # 1H 방향 확인
    h1_ema_period: int = 20
    # 5M 진입 타이밍
    m5_rsi_period: int = 14
    m5_rsi_overbought: float = 70.0
    m5_rsi_oversold: float = 30.0
    m5_bb_period: int = 20
    m5_bb_std: float = 2.0
    # 거래량 필터
    volume_multiplier: float = 1.5
    volume_lookback: int = 20
    # 변동성 킬스위치
    volatility_atr_kill_multiplier: float = 3.0
    volatility_atr_lookback: int = 50


class StrategyConfig(BaseModel):
    """Strategy enablement and thresholds."""

    enabled: dict[str, bool] = Field(
        default_factory=lambda: {
            "break_retest": True,
            "liquidity_raid": True,
            "fair_value_gap": True,
            "order_block": True,
            "choch": True,
        }
    )
    min_signal_score: float = 0.55
    range_middle_exclusion: float = 0.2
    trend_weight: float = 1.2
    raid_weight: float = 1.15
    breakout_weight: float = 1.1
    momentum_weight: float = 1.12
    max_chase_atr_multiple: float = 0.35
    retest_tolerance_atr: float = 0.4
    confirmation_volume_multiple: float = 1.15
    merge_nearby_target_threshold_pct: float = 0.0015
    reject_trade_if_targets_are_inside_range_middle: bool = True


class StrategyRouterConfig(BaseModel):
    """Conflict-resolution and regime-router settings."""

    max_active_strategy_groups: int = 4
    max_candidate_signals_per_symbol: int = 3
    max_primary_signals_per_symbol: int = 1
    regime_router_enabled: bool = True
    signal_conflict_resolution_enabled: bool = True
    strategy_cooldown_minutes: float = 15
    symbol_cooldown_minutes: float = 10
    reject_overlapping_strategy_signals: bool = True
    merge_same_direction_signals: bool = True
    block_opposite_signals_same_window: bool = True
    same_direction_overlap_threshold: float = 0.68
    opposite_signal_score_gap: float = 0.08


class EVConfig(BaseModel):
    """Expected-value filter configuration."""

    min_ev: float = 0.15
    min_expected_r: float = 2.0
    min_historical_win_rate: float = 0.42
    estimated_maker_fee_bps: float = 2.0
    estimated_taker_fee_bps: float = 6.0
    slippage_bps: float = 2.5
    news_penalty_high: float = 0.2
    funding_penalty_minutes: int = 20
    market_impact_penalty_bps: float = 1.5
    min_rr_to_tp1_break_retest: float = 0.4
    preferred_rr_to_tp2_break_retest: float = 1.2
    min_rr_to_tp1_liquidity_raid: float = 0.5
    preferred_rr_to_tp2_liquidity_raid: float = 1.5
    min_rr_to_tp1_fair_value_gap: float = 0.4
    preferred_rr_to_tp2_fair_value_gap: float = 1.2
    min_rr_to_tp1_order_block: float = 0.4
    preferred_rr_to_tp2_order_block: float = 1.2
    min_rr_to_tp1_choch: float = 0.5
    preferred_rr_to_tp2_choch: float = 1.5


class RiskConfig(BaseModel):
    """Risk engine controls."""

    stop_mode: str = "structure"
    target_mode: str = "structure"
    use_atr_buffer_for_stop: bool = True
    atr_buffer_multiplier: float = 0.15
    min_stop_distance_pct: float = 0.002
    max_stop_distance_pct: float = 0.02
    enable_structural_tp1_tp2_tp3: bool = True
    # 포지션 사이징 모드: "risk_based" (손절거리 기반) | "equity_share" (자산 균등 배분)
    sizing_mode: str = "equity_share"  # equity_share | risk_based | atr_proportional | kelly
    kelly_fraction: float = 0.25       # Kelly 분율 (0.25 = quarter Kelly)
    atr_stop_multiplier: float = 1.5   # ATR × 이 값 = 손절 거리
    # equity_share 모드: 총 자산을 몇 개 포지션으로 나눌지 (증거금 기준)
    equity_share_count: int = 10
    risk_per_trade: float = 0.005
    min_risk_per_trade: float = 0.0025
    max_risk_per_trade: float = 0.01
    max_concurrent_positions: int = 3      # 최대 동시 포지션 (10→3)
    # 포트폴리오 히트 한도: 전체 증거금 / 총자산 (equity_share 모드에서도 안전망)
    max_portfolio_heat_pct: float = 1.0
    max_daily_loss_r: float = 3.0
    max_daily_loss_pct: float = 2.0        # 일일 최대 손실 % → 당일 중단
    max_consecutive_losses: int = 3         # 연속 손실 → 쿨다운 (4→3)
    cooldown_minutes_after_loss: int = 30
    # 서킷 브레이커 토글
    daily_loss_limit_enabled: bool = True
    consecutive_loss_cooldown_enabled: bool = True
    drawdown_limit_enabled: bool = True
    kill_switch_enabled: bool = True
    # 같은 손절 이유 연속 발동 시 쿨다운
    stop_reason_cooldown_threshold: int = 5       # N회 연속 같은 이유 손절 시 쿨다운
    stop_reason_cooldown_minutes: int = 30        # 쿨다운 지속 시간(분)
    max_daily_orders: int = 10             # 일일 최대 거래 (40→10)
    max_position_hold_minutes: int = 30
    stale_eviction_minutes: int = 30  # 새 진입 시 이 시간 초과 포지션 먼저 정리
    max_account_drawdown_pct: float = 8.0
    kill_switch_unrealized_loss_pct: float = 4.0
    allow_multiple_positions_per_symbol: bool = False
    funding_block_before_minutes: int = 15
    funding_block_after_minutes: int = 10
    event_block_before_minutes: int = 20
    event_block_after_minutes: int = 20
    partial_tp_ratio: float = 0.5
    tp1_partial_close_pct: float = 0.5
    tp2_partial_close_pct: float = 0.25
    tp3_partial_close_pct: float = 0.15
    move_sl_to_be_after_tp1: bool = True
    break_even_buffer_r: float = 0.2
    be_offset_r: float = 0.1
    trailing_stop_enabled: bool = True
    hard_stop_loss_pct: float | None = None

    @field_validator("risk_per_trade")
    @classmethod
    def validate_risk_range(cls, value: float) -> float:
        """Keep default risk within allowed bounds."""

        if not 0 < value <= 0.02:
            raise ValueError("risk_per_trade must be between 0 and 0.02")
        return value


class ExecutionConfig(BaseModel):
    """Execution-router options."""

    maker_first: bool = True
    taker_fallback: bool = True
    taker_fallback_volatility_threshold: float = 0.9
    max_requote_attempts: int = 2
    maker_timeout_seconds: int = 15
    split_entry_count: int = 1
    split_exit_count: int = 2
    use_exchange_plan_orders: bool = False
    default_leverage: float = 20.0
    per_strategy_leverage: dict[str, float] = Field(
        default_factory=lambda: {
            "break_retest": 25.0,      # 안정적 패턴 — 중간 레버리지
            "liquidity_raid": 40.0,    # 고EV 빠른 이동 — 높은 레버리지
            "fair_value_gap": 30.0,    # 기관 구간 — 중상위 레버리지
            "order_block": 25.0,       # 역방향 반전 — 중간 레버리지
            "choch": 20.0,             # 추세전환 — 최소 레버리지 (불확실성 높음)
        }
    )


class UniverseConfig(BaseModel):
    """Universe selection options."""

    active_universe_size: int = 50
    include_btc_eth_always: bool = True
    min_24h_quote_volume: float = 2_000_000.0
    max_spread_bps: float = 8.0
    min_listing_age_hours: int = 48
    refresh_seconds: int = 60
    instant_volume_spike_multiplier: float = 2.0


class NewsConfig(BaseModel):
    """News engine options."""

    enabled: bool = True
    poll_seconds: int = 180
    ai_analysis_mode: str = "important_only"
    ai_cooldown_minutes_after_failure: int = 180
    daily_summary_hour_kst: int = 8
    high_impact_alerts: bool = True
    duplicate_cache_hours: int = 24
    block_high_impact_entries: bool = True
    max_items_per_source: int = 20


class TelegramConfig(BaseModel):
    """Telegram feature configuration."""

    enabled: bool = True
    admin_ids: list[int] = Field(default_factory=list)
    poll_seconds: int = 5
    send_daily_summary: bool = True
    send_weekly_summary: bool = True
    allow_close_commands: bool = False
    startup_alert_enabled: bool = True
    startup_alert_timeout_seconds: int = 15
    heartbeat_alert_enabled: bool = False
    heartbeat_minutes: int = 60


class DashboardConfig(BaseModel):
    """Streamlit dashboard options."""

    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8501
    refresh_seconds: int = 15


class DatabaseConfig(BaseModel):
    """Database configuration."""

    sqlite_path: str = "data/trading_bot.sqlite3"

    @classmethod
    def resolve_sqlite_path(cls) -> str:
        """Railway Volume이 마운트된 경우 /data 경로 우선 사용."""
        import os
        data_dir = os.environ.get("DATA_DIR", "").strip()
        if data_dir:
            return f"{data_dir.rstrip('/')}/trading_bot.sqlite3"
        return "data/trading_bot.sqlite3"


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    directory: str = "logs"


class ExchangeConfig(BaseModel):
    """Bitget connection settings."""

    base_url: str = "https://api.bitget.com"
    ws_public_url: str = "wss://ws.bitget.com/v2/ws/public"
    ws_private_url: str = "wss://ws.bitget.com/v2/ws/private"
    demo_header: str = "1"
    product_types: list[ProductType] = Field(
        default_factory=lambda: [
            ProductType.USDT_FUTURES,
            ProductType.USDC_FUTURES,
            # ProductType.COIN_FUTURES,  # ETH 마진 없어서 항상 실패 → 비활성화
        ]
    )
    request_timeout_seconds: int = 15
    max_retries: int = 5
    recv_window_ms: int = 30_000
    live_confirmation_required: bool = True
    live_streamlit_confirmed: bool = False


class RuntimeConfig(BaseModel):
    """Runtime loop settings."""

    scan_interval_seconds: int = 20
    healthcheck_seconds: int = 30
    state_file: str = "state/runtime_state.json"
    graceful_shutdown_seconds: int = 20


class EnvSecrets(BaseSettings):
    """Secret settings sourced from .env or OS environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH) if ENV_PATH.exists() else None,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    bitget_api_key: str = ""
    bitget_api_secret: str = ""
    bitget_api_passphrase: str = ""
    bitget_demo_api_key: str = ""
    bitget_demo_api_secret: str = ""
    bitget_demo_api_passphrase: str = ""
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""           # AI 듀얼 프로바이더 — 에스컬레이션용
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    live_trading_enabled: bool = False

    def model_post_init(self, __context: Any) -> None:
        """os.environ 직접 읽기 fallback (Railway 등 일부 환경 대응)."""
        import os
        if not self.bitget_demo_api_key:
            object.__setattr__(self, "bitget_demo_api_key", os.environ.get("BITGET_DEMO_API_KEY", ""))
        if not self.bitget_demo_api_secret:
            object.__setattr__(self, "bitget_demo_api_secret", os.environ.get("BITGET_DEMO_API_SECRET", ""))
        if not self.bitget_demo_api_passphrase:
            object.__setattr__(self, "bitget_demo_api_passphrase", os.environ.get("BITGET_DEMO_API_PASSPHRASE", ""))
        if not self.bitget_api_key:
            object.__setattr__(self, "bitget_api_key", os.environ.get("BITGET_API_KEY", ""))
        if not self.bitget_api_secret:
            object.__setattr__(self, "bitget_api_secret", os.environ.get("BITGET_API_SECRET", ""))
        if not self.bitget_api_passphrase:
            object.__setattr__(self, "bitget_api_passphrase", os.environ.get("BITGET_API_PASSPHRASE", ""))
        if not self.openai_api_key:
            object.__setattr__(self, "openai_api_key", os.environ.get("OPENAI_API_KEY", ""))
        if not self.telegram_bot_token:
            object.__setattr__(self, "telegram_bot_token", os.environ.get("TELEGRAM_BOT_TOKEN", ""))


class AppSettings(BaseModel):
    """Full runtime configuration."""

    mode: TradingMode = TradingMode.DEMO
    timeframes: TimeframeConfig = Field(default_factory=TimeframeConfig)
    mtf: MTFConfig = Field(default_factory=MTFConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    strategy_router: StrategyRouterConfig = Field(default_factory=StrategyRouterConfig)
    ev: EVConfig = Field(default_factory=EVConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    news: NewsConfig = Field(default_factory=NewsConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    secrets: EnvSecrets = Field(default_factory=EnvSecrets)

    @property
    def db_path(self) -> Path:
        """Return absolute SQLite path. Railway Volume(DATA_DIR) 우선."""
        import os
        data_dir = os.environ.get("DATA_DIR", "").strip()
        if data_dir:
            return Path(data_dir) / "trading_bot.sqlite3"
        return ROOT_DIR / self.database.sqlite_path

    @property
    def state_path(self) -> Path:
        """Return absolute state JSON path."""

        return ROOT_DIR / self.runtime.state_file

    def ensure_live_trading_allowed(self) -> None:
        """Validate non-negotiable live-trading safety checks."""

        if self.mode != TradingMode.LIVE:
            return
        if not self.secrets.live_trading_enabled:
            raise ValueError("LIVE mode requires LIVE_TRADING_ENABLED=true in .env")
        if not self.exchange.live_confirmation_required:
            raise ValueError("LIVE mode requires live_confirmation_required=true")
        if not self.exchange.live_streamlit_confirmed:
            raise ValueError("LIVE mode requires final Streamlit confirmation")

    def to_runtime_dict(self) -> dict[str, Any]:
        """Serialize without secrets when saving runtime settings."""

        payload = self.model_dump(mode="json")
        payload.pop("secrets", None)
        return payload


class SettingsManager:
    """Load, validate, and persist settings."""

    def __init__(
        self,
        defaults_path: Path = DEFAULTS_PATH,
        bot_settings_path: Path = BOT_SETTINGS_PATH,
    ) -> None:
        self.defaults_path = defaults_path
        self.bot_settings_path = bot_settings_path
        self._last_mtime: float | None = None

    def load(self) -> AppSettings:
        """Load settings from defaults, overrides, and environment."""

        defaults = load_json(self.defaults_path, default={})
        overrides = load_json(self.bot_settings_path, default={})
        merged = deep_merge(defaults, overrides)
        merged = self._normalize_strategy_settings(merged)
        try:
            settings = AppSettings(**merged, secrets=EnvSecrets())
        except ValidationError as exc:
            raise ValueError(f"설정 로딩 실패: {exc}") from exc
        settings.ensure_live_trading_allowed()
        if self.bot_settings_path.exists():
            self._last_mtime = self.bot_settings_path.stat().st_mtime
        return settings

    def save(self, settings: AppSettings | dict[str, Any]) -> None:
        """Persist UI-editable settings to bot_settings.json."""

        if isinstance(settings, AppSettings):
            payload = settings.to_runtime_dict()
        else:
            payload = settings
        dump_json(self.bot_settings_path, payload)
        self._last_mtime = self.bot_settings_path.stat().st_mtime

    def reload_if_changed(self) -> AppSettings | None:
        """Reload settings if bot_settings.json changed."""

        if not self.bot_settings_path.exists():
            return None
        mtime = self.bot_settings_path.stat().st_mtime
        if self._last_mtime is None or mtime > self._last_mtime:
            self._last_mtime = mtime
            return self.load()
        return None

    def create_initial_files(self) -> None:
        """Create defaults and bot_settings files when absent."""

        if not self.defaults_path.exists():
            dump_json(self.defaults_path, AppSettings().to_runtime_dict())
        if not self.bot_settings_path.exists():
            settings = AppSettings()
            dump_json(self.bot_settings_path, settings.to_runtime_dict())

    def _normalize_strategy_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Support both legacy nested strategy settings and flat compatibility keys."""

        normalized = dict(payload)
        strategy = dict(normalized.get("strategy") or {})
        enabled = dict(strategy.get("enabled") or {})
        router = dict(normalized.get("strategy_router") or strategy.get("router") or {})

        flat_enabled_map = {
            "strategy_vincent_enabled": "break_retest",
            "strategy_liquidity_reclaim_enabled": "liquidity_raid",
        }
        for flat_key, strategy_key in flat_enabled_map.items():
            if flat_key in normalized:
                enabled[strategy_key] = bool(normalized.get(flat_key))

        # 유효한 전략명만 허용 (구버전 session_breakout, momentum_pullback 자동 제거)
        _VALID_STRATEGY_NAMES = {"break_retest", "liquidity_raid", "fair_value_gap", "order_block", "choch"}
        enabled = {k: v for k, v in enabled.items() if k in _VALID_STRATEGY_NAMES}

        flat_router_keys = {
            "max_active_strategy_groups",
            "max_candidate_signals_per_symbol",
            "max_primary_signals_per_symbol",
            "regime_router_enabled",
            "signal_conflict_resolution_enabled",
            "strategy_cooldown_minutes",
            "symbol_cooldown_minutes",
            "reject_overlapping_strategy_signals",
            "merge_same_direction_signals",
            "block_opposite_signals_same_window",
        }
        for key in flat_router_keys:
            if key in normalized:
                router[key] = normalized.get(key)

        strategy["enabled"] = enabled
        strategy["router"] = router
        normalized["strategy"] = strategy
        normalized["strategy_router"] = router
        return normalized


def getenv_bool(name: str, default: bool = False) -> bool:
    """Helper for shell-side toggles."""

    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_settings_snapshot(settings: AppSettings) -> dict[str, Any]:
    """Return metadata used in journal and Telegram status messages."""

    return {
        "mode": settings.mode.value,
        "active_universe_size": settings.universe.active_universe_size,
        "risk_per_trade": settings.risk.risk_per_trade,
        "maker_first": settings.execution.maker_first,
        "enabled_strategies": [name for name, enabled in settings.strategy.enabled.items() if enabled],
        "regime_router_enabled": settings.strategy_router.regime_router_enabled,
        "telegram_enabled": settings.telegram.enabled,
        "streamlit_enabled": settings.dashboard.enabled,
        "loaded_at": utc_now().isoformat(timespec="seconds"),
    }
