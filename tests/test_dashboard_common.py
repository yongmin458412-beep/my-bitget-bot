"""Tests for dashboard control helpers."""

from __future__ import annotations

from pathlib import Path

from dashboard.common import enqueue_control_command, load_control_commands


def test_enqueue_control_command_creates_pending_item(tmp_path: Path) -> None:
    """Dashboard actions should be persisted as pending commands."""

    command_path = tmp_path / "control_commands.json"
    command = enqueue_control_command(
        "close_all_positions",
        payload={"source": "test"},
        path=command_path,
    )

    snapshot = load_control_commands(command_path)
    assert snapshot["pending"][0]["id"] == command["id"]
    assert snapshot["pending"][0]["action"] == "close_all_positions"
    assert snapshot["pending"][0]["payload"]["source"] == "test"
    assert snapshot["history"] == []
