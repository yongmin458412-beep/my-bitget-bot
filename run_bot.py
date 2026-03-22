"""Entrypoint for the async trading bot."""

from __future__ import annotations

import asyncio
import fcntl
import os
import signal
import sys
from pathlib import Path

from app.main import TradingApplication

LOCK_FILE = Path("/tmp/bitget_bot.lock")


def _acquire_lock() -> int:
    """Acquire an exclusive PID lock. Exit if another instance is running."""
    fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_WRONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        try:
            existing_pid = LOCK_FILE.read_text().strip()
        except Exception:
            existing_pid = "unknown"
        print(
            f"❌ 봇이 이미 실행 중입니다 (PID: {existing_pid}).\n"
            "중복 실행을 방지합니다. 기존 프로세스를 먼저 종료하세요.\n"
            f"  kill {existing_pid}",
            file=sys.stderr,
        )
        os.close(fd)
        sys.exit(1)

    LOCK_FILE.write_text(str(os.getpid()))
    return fd


def _release_lock(fd: int) -> None:
    """Release the PID lock file."""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass


async def runner() -> None:
    """Run the trading application until interrupted."""

    app = TradingApplication()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(app.request_shutdown(f"signal:{s.name}")))

    await app.start()
    try:
        await app.stop_event.wait()
    finally:
        await app.stop()


if __name__ == "__main__":
    lock_fd = _acquire_lock()
    print(f"✅ 봇 시작 (PID: {os.getpid()})", flush=True)
    try:
        asyncio.run(runner())
    finally:
        _release_lock(lock_fd)
