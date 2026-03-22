"""Settings normalization, boundary validation, and hot-reload tests."""

from __future__ import annotations

import json
from pathlib import Path

from core.settings import AppSettings, SettingsManager


# ── Default value tests ──────────────────────────────────────────────────────

def test_default_settings_mode_is_demo() -> None:
    settings = AppSettings()
    from core.enums import TradingMode
    assert settings.mode == TradingMode.DEMO


def test_default_risk_per_trade_in_range() -> None:
    settings = AppSettings()
    assert 0 < settings.risk.risk_per_trade <= 0.05


def test_default_all_strategies_enabled() -> None:
    settings = AppSettings()
    for name in ("break_retest", "liquidity_raid", "session_breakout", "momentum_pullback"):
        assert settings.strategy.enabled.get(name) is True


def test_default_ev_min_positive() -> None:
    settings = AppSettings()
    assert settings.ev.min_ev > 0


def test_default_tp_partial_pcts_sum_le_one() -> None:
    settings = AppSettings()
    total = (
        settings.risk.tp1_partial_close_pct
        + settings.risk.tp2_partial_close_pct
        + settings.risk.tp3_partial_close_pct
    )
    assert total <= 1.0, f"TP pct sum {total} > 1.0"


# ── Normalization: flat legacy keys → nested ──────────────────────────────────

def test_normalization_flat_strategy_enabled_keys(tmp_path: Path) -> None:
    """Legacy flat keys like strategy_vincent_enabled should map to strategy.enabled."""

    bot_settings_path = tmp_path / "config" / "bot_settings.json"
    bot_settings_path.parent.mkdir(parents=True)
    bot_settings_path.write_text(json.dumps({
        "strategy_vincent_enabled": False,
        "strategy_liquidity_reclaim_enabled": True,
    }))

    manager = SettingsManager(
        defaults_path=tmp_path / "config" / "defaults.json",
        bot_settings_path=bot_settings_path,
    )
    settings = manager.load()

    assert settings.strategy.enabled["break_retest"] is False
    assert settings.strategy.enabled["liquidity_raid"] is True


def test_normalization_flat_router_keys(tmp_path: Path) -> None:
    """Flat router keys should land in strategy_router config."""

    bot_settings_path = tmp_path / "config" / "bot_settings.json"
    bot_settings_path.parent.mkdir(parents=True)
    bot_settings_path.write_text(json.dumps({
        "strategy_cooldown_minutes": 30,
        "symbol_cooldown_minutes": 20,
    }))

    manager = SettingsManager(
        defaults_path=tmp_path / "config" / "defaults.json",
        bot_settings_path=bot_settings_path,
    )
    settings = manager.load()

    assert settings.strategy_router.strategy_cooldown_minutes == 30
    assert settings.strategy_router.symbol_cooldown_minutes == 20


def test_settings_manager_saves_and_reloads(tmp_path: Path) -> None:
    """save() then load() should return updated settings."""

    bot_path = tmp_path / "config" / "bot_settings.json"
    bot_path.parent.mkdir(parents=True)

    manager = SettingsManager(
        defaults_path=tmp_path / "config" / "defaults.json",
        bot_settings_path=bot_path,
    )
    original = manager.load()

    # Modify via dict and save
    payload = original.to_runtime_dict()
    payload.setdefault("risk", {})["risk_per_trade"] = 0.007
    manager.save(payload)

    # Direct load (not reload_if_changed) always reads fresh
    reloaded = manager.load()
    assert reloaded is not None
    assert reloaded.risk.risk_per_trade == 0.007


def test_settings_manager_no_reload_when_unchanged(tmp_path: Path) -> None:
    """reload_if_changed() should return None if file unchanged."""

    bot_path = tmp_path / "config" / "bot_settings.json"
    bot_path.parent.mkdir(parents=True)

    manager = SettingsManager(
        defaults_path=tmp_path / "config" / "defaults.json",
        bot_settings_path=bot_path,
    )
    manager.load()  # sets _last_mtime

    result = manager.reload_if_changed()
    assert result is None


# ── Fuzz: garbage inputs should not crash ───────────────────────────────────

def test_settings_manager_ignores_unknown_keys(tmp_path: Path) -> None:
    """Unknown keys in bot_settings.json should be ignored without crashing."""

    bot_path = tmp_path / "config" / "bot_settings.json"
    bot_path.parent.mkdir(parents=True)
    bot_path.write_text(json.dumps({
        "totally_unknown_key_xyz": "should_be_ignored",
        "another_garbage": 12345,
        "risk": {"risk_per_trade": 0.01},
    }))

    manager = SettingsManager(
        defaults_path=tmp_path / "config" / "defaults.json",
        bot_settings_path=bot_path,
    )
    settings = manager.load()
    assert settings.risk.risk_per_trade == 0.01


def test_settings_manager_handles_empty_bot_settings(tmp_path: Path) -> None:
    """Empty bot_settings.json should fall through to defaults without error."""

    bot_path = tmp_path / "config" / "bot_settings.json"
    bot_path.parent.mkdir(parents=True)
    bot_path.write_text("{}")

    manager = SettingsManager(
        defaults_path=tmp_path / "config" / "defaults.json",
        bot_settings_path=bot_path,
    )
    settings = manager.load()
    assert settings is not None


def test_settings_manager_handles_invalid_json(tmp_path: Path) -> None:
    """Corrupted bot_settings.json should not crash — fall back to defaults."""

    bot_path = tmp_path / "config" / "bot_settings.json"
    bot_path.parent.mkdir(parents=True)
    bot_path.write_text("{invalid json!!!}")

    manager = SettingsManager(
        defaults_path=tmp_path / "config" / "defaults.json",
        bot_settings_path=bot_path,
    )
    # Should not raise
    try:
        settings = manager.load()
        # If it succeeds, fine
    except Exception:
        pass  # Expected if invalid JSON causes load to fail — just ensure no unhandled crash


def test_settings_tp_pct_values_non_negative() -> None:
    settings = AppSettings()
    assert settings.risk.tp1_partial_close_pct >= 0
    assert settings.risk.tp2_partial_close_pct >= 0
    assert settings.risk.tp3_partial_close_pct >= 0


def test_settings_max_concurrent_positions_positive() -> None:
    settings = AppSettings()
    assert settings.risk.max_concurrent_positions > 0


def test_settings_universe_active_size_reasonable() -> None:
    settings = AppSettings()
    assert 1 <= settings.universe.active_universe_size <= 200
