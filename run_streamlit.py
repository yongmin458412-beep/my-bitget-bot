"""Launch the Streamlit dashboard with configured host/port."""

from __future__ import annotations

import subprocess
import sys

from core.settings import SettingsManager


def main() -> None:
    """Run the Streamlit control panel."""

    settings = SettingsManager().load()
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "dashboard/streamlit_app.py",
        "--server.address",
        settings.dashboard.host,
        "--server.port",
        str(settings.dashboard.port),
    ]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()

