"""Settings tests."""

from __future__ import annotations

import json
from pathlib import Path

from core.settings import SettingsManager


def test_settings_default_mode_demo(tmp_path: Path) -> None:
    """Settings manager should load DEMO by default."""

    defaults = tmp_path / "defaults.json"
    bot_settings = tmp_path / "bot_settings.json"
    defaults.write_text(json.dumps({"mode": "DEMO"}), encoding="utf-8")
    bot_settings.write_text("{}", encoding="utf-8")
    manager = SettingsManager(defaults_path=defaults, bot_settings_path=bot_settings)
    settings = manager.load()
    assert settings.mode.value == "DEMO"


def test_settings_default_min_expected_r_is_two(tmp_path: Path) -> None:
    """Default EV config should target 1:2 risk-reward."""

    defaults = tmp_path / "defaults.json"
    bot_settings = tmp_path / "bot_settings.json"
    defaults.write_text(json.dumps({"mode": "DEMO", "ev": {"min_expected_r": 2.0}}), encoding="utf-8")
    bot_settings.write_text("{}", encoding="utf-8")
    manager = SettingsManager(defaults_path=defaults, bot_settings_path=bot_settings)
    settings = manager.load()
    assert settings.ev.min_expected_r == 2.0


def test_settings_load_structural_defaults(tmp_path: Path) -> None:
    """Structural stop/target defaults should load through nested config sections."""

    defaults = tmp_path / "defaults.json"
    bot_settings = tmp_path / "bot_settings.json"
    defaults.write_text(
        json.dumps(
            {
                "mode": "DEMO",
                "risk": {
                    "stop_mode": "structure",
                    "target_mode": "structure",
                    "atr_buffer_multiplier": 0.15,
                    "tp1_partial_close_pct": 0.5,
                    "tp2_partial_close_pct": 0.25,
                    "tp3_partial_close_pct": 0.15,
                    "be_offset_r": 0.1,
                },
                "ev": {
                    "min_rr_to_tp1_break_retest": 1.3,
                    "preferred_rr_to_tp2_break_retest": 1.8,
                    "min_rr_to_tp1_liquidity_raid": 1.5,
                    "preferred_rr_to_tp2_liquidity_raid": 2.0,
                    "min_rr_to_tp1_session_breakout": 1.35,
                    "preferred_rr_to_tp2_session_breakout": 2.0,
                    "min_rr_to_tp1_momentum_pullback": 1.2,
                    "preferred_rr_to_tp2_momentum_pullback": 1.7,
                },
                "execution": {
                    "default_leverage": 20.0,
                    "per_strategy_leverage": {
                        "break_retest": 20.0,
                        "liquidity_raid": 20.0,
                        "session_breakout": 25.0,
                        "momentum_pullback": 30.0,
                    },
                },
                "strategy": {
                    "merge_nearby_target_threshold_pct": 0.0015,
                    "reject_trade_if_targets_are_inside_range_middle": True,
                },
            }
        ),
        encoding="utf-8",
    )
    bot_settings.write_text("{}", encoding="utf-8")
    manager = SettingsManager(defaults_path=defaults, bot_settings_path=bot_settings)
    settings = manager.load()
    assert settings.risk.stop_mode == "structure"
    assert settings.risk.target_mode == "structure"
    assert settings.risk.tp1_partial_close_pct == 0.5
    assert settings.risk.tp2_partial_close_pct == 0.25
    assert settings.risk.tp3_partial_close_pct == 0.15
    assert settings.risk.be_offset_r == 0.1
    assert settings.ev.min_rr_to_tp1_break_retest == 1.3
    assert settings.ev.preferred_rr_to_tp2_liquidity_raid == 2.0
    assert settings.ev.min_rr_to_tp1_session_breakout == 1.35
    assert settings.ev.preferred_rr_to_tp2_momentum_pullback == 1.7
    assert settings.execution.default_leverage == 20.0
    assert settings.execution.per_strategy_leverage["session_breakout"] == 25.0
    assert settings.execution.per_strategy_leverage["momentum_pullback"] == 30.0
    assert settings.strategy.merge_nearby_target_threshold_pct == 0.0015
    assert settings.strategy.reject_trade_if_targets_are_inside_range_middle is True


def test_settings_flat_strategy_router_keys_are_normalized(tmp_path: Path) -> None:
    """Legacy flat strategy flags should hydrate the nested strategy/router config."""

    defaults = tmp_path / "defaults.json"
    bot_settings = tmp_path / "bot_settings.json"
    defaults.write_text(
        json.dumps(
            {
                "mode": "DEMO",
                "strategy_vincent_enabled": True,
                "strategy_liquidity_reclaim_enabled": True,
                "strategy_session_breakout_enabled": False,
                "strategy_momentum_pullback_enabled": True,
                "max_active_strategy_groups": 4,
                "max_candidate_signals_per_symbol": 3,
                "max_primary_signals_per_symbol": 1,
                "regime_router_enabled": True,
                "signal_conflict_resolution_enabled": True,
                "strategy_cooldown_minutes": 15,
                "symbol_cooldown_minutes": 10,
                "reject_overlapping_strategy_signals": True,
                "merge_same_direction_signals": True,
                "block_opposite_signals_same_window": True,
            }
        ),
        encoding="utf-8",
    )
    bot_settings.write_text("{}", encoding="utf-8")
    settings = SettingsManager(defaults_path=defaults, bot_settings_path=bot_settings).load()

    assert settings.strategy.enabled["break_retest"] is True
    assert settings.strategy.enabled["liquidity_raid"] is True
    assert settings.strategy.enabled["session_breakout"] is False
    assert settings.strategy.enabled["momentum_pullback"] is True
    assert settings.strategy_router.max_active_strategy_groups == 4
    assert settings.strategy_router.regime_router_enabled is True
    assert settings.strategy_router.block_opposite_signals_same_window is True
