"""Small readiness endpoint for an idle calibration container."""

import json
import time
from pathlib import Path

from aiohttp import web

from charavoice_engine.scripts.calibrate import _gpu_info


async def status(_request: web.Request) -> web.Response:
    return web.json_response({"ready": True, "mode": "calibration-idle"})


def main() -> None:
    started = time.time()
    record = {
        "ready": True,
        "start_epoch": started,
        "ready_seconds": 0.0,
        "gpu": _gpu_info(),
        "cache_mode": "calibration-idle",
    }
    runs = Path("/app/runs")
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "startup.json").write_text(json.dumps(record, indent=2) + "\n")
    app = web.Application()
    app.router.add_get("/status", status)
    web.run_app(app, host=["0.0.0.0", "::"], port=5000, access_log=None)


if __name__ == "__main__":
    main()
