"""Root entrypoint for Streamlit Community Cloud and local execution.

Automatically launches the embedded FastAPI microservice in a background daemon thread
(if port 8000 is not already bound) and renders the multi-page IDFS dashboard.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

# Add src to sys.path
root_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def is_port_in_use(port: int = 8000) -> bool:
    """Check if port 8000 is already actively bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def ensure_fastapi_running() -> None:
    """Launch embedded FastAPI ASGI server in background thread if not already active."""
    if not is_port_in_use(8000):
        try:
            import uvicorn

            from idfs.api.main import app

            def run_server() -> None:
                config = uvicorn.Config(
                    app,
                    host="127.0.0.1",
                    port=8000,
                    log_level="warning",
                    access_log=False,
                )
                server = uvicorn.Server(config)
                server.run()

            thread = threading.Thread(target=run_server, daemon=True, name="FastAPI-Daemon")
            thread.start()
            time.sleep(1.0)
        except Exception:
            pass


ensure_fastapi_running()

# Launch Streamlit Main App
from idfs.app.main import main  # noqa: E402

if __name__ == "__main__":
    main()
