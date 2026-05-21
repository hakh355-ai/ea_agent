"""
Watchdog — monitors the bridge and restarts it automatically if it goes down.

How it works:
  - Checks GET /health every 60 seconds
  - If bridge doesn't respond: starts it via uvicorn
  - Logs every restart to .tmp/watchdog.log
  - Runs forever; designed to start on Windows boot via Task Scheduler

Setup (one-time):
  1. Open Task Scheduler → Create Basic Task
  2. Trigger: At system startup
  3. Action: Start a program
     Program: python
     Arguments: C:\\Users\\khali\\Documents\\EA_Agent\\watchdog.py
     Start in: C:\\Users\\khali\\Documents\\EA_Agent
  4. General: Run whether user is logged on or not

Usage:
  python watchdog.py
"""
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LOG_PATH   = Path(".tmp/watchdog.log")
BRIDGE_URL = f"http://{os.getenv('BRIDGE_HOST','127.0.0.1')}:{os.getenv('BRIDGE_PORT','5000')}"
CHECK_INTERVAL = 60   # seconds between health checks
STARTUP_GRACE  = 10   # seconds to wait after launching before first check

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WATCHDOG] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

_bridge_proc: subprocess.Popen | None = None


def _log(msg: str):
    logger.info(msg)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _bridge_healthy() -> bool:
    try:
        import urllib.request
        with urllib.request.urlopen(f"{BRIDGE_URL}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def _start_bridge():
    global _bridge_proc
    _log("Starting bridge...")
    try:
        _bridge_proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "bridge.server:app",
                "--host", os.getenv("BRIDGE_HOST", "127.0.0.1"),
                "--port", os.getenv("BRIDGE_PORT", "5000"),
            ],
            cwd=Path(__file__).parent,
            stdout=open(".tmp/bridge.log", "a"),
            stderr=subprocess.STDOUT,
        )
        _log(f"Bridge process started (PID {_bridge_proc.pid})")
        time.sleep(STARTUP_GRACE)
    except Exception as e:
        _log(f"Failed to start bridge: {e}")


def _is_proc_running() -> bool:
    return _bridge_proc is not None and _bridge_proc.poll() is None


def main():
    _log(f"Watchdog started. Monitoring {BRIDGE_URL} every {CHECK_INTERVAL}s")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    restart_count = 0

    # Start bridge immediately on first run
    if not _bridge_healthy():
        _start_bridge()

    while True:
        time.sleep(CHECK_INTERVAL)

        if _bridge_healthy():
            continue

        # Bridge is down
        restart_count += 1
        _log(f"Bridge unreachable — restart #{restart_count}")

        # Kill stale process if still running
        if _is_proc_running():
            try:
                _bridge_proc.terminate()
                _bridge_proc.wait(timeout=5)
            except Exception:
                pass

        _start_bridge()

        if _bridge_healthy():
            _log(f"Bridge recovered after restart #{restart_count}")
        else:
            _log(f"Bridge still unreachable after restart #{restart_count} — will retry in {CHECK_INTERVAL}s")


if __name__ == "__main__":
    main()
