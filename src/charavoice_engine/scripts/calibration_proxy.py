"""Local HTTP router used while comparing one or more TTS workers.

The routing policy mirrors the production OGG setup: non-streaming requests
use the first worker, while streaming requests rotate across all workers.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import itertools
import json

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

_HOP_HEADERS = {"connection", "content-length", "transfer-encoding", "content-encoding", "keep-alive"}


def create_app(backends: list[str]) -> web.Application:
    if not backends:
        raise ValueError("At least one backend is required")
    bases = [backend.rstrip("/") for backend in backends]
    app = web.Application(client_max_size=16 * 1024 * 1024)
    app["next_backend"] = itertools.count()

    async def client_context(application: web.Application):
        application["client"] = ClientSession(connector=TCPConnector(limit=0), timeout=ClientTimeout(total=None))
        yield
        await application["client"].close()

    app.cleanup_ctx.append(client_context)

    async def cluster_status(request: web.Request) -> web.Response:
        async def fetch(base: str):
            try:
                async with request.app["client"].get(base + "/status", timeout=5) as response:
                    return response.status, await response.json()
            except (TimeoutError, OSError, ValueError) as exc:
                return 503, {"error": str(exc)}

        results = await asyncio.gather(*(fetch(base) for base in bases))
        healthy = all(status == 200 for status, _ in results)
        return web.json_response(
            {"healthy": healthy, "workers": [body for _, body in results]},
            status=200 if healthy else 503,
        )

    async def proxy(request: web.Request) -> web.StreamResponse:
        body = await request.read()
        streaming = False
        if request.path == "/voice" and request.method == "POST":
            with contextlib.suppress(ValueError, AttributeError):
                streaming = json.loads(body).get("streaming") is True
        index = next(request.app["next_backend"]) % len(bases) if streaming else 0
        target = bases[index] + request.rel_url.path_qs
        headers = {"Content-Type": request.headers.get("Content-Type", "application/json")}
        try:
            async with request.app["client"].request(request.method, target, data=body, headers=headers) as upstream:
                response = web.StreamResponse(
                    status=upstream.status,
                    headers={k: v for k, v in upstream.headers.items() if k.lower() not in _HOP_HEADERS},
                )
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
                return response
        except (TimeoutError, OSError) as exc:
            raise web.HTTPBadGateway(text=f"backend {index} unavailable: {exc}") from exc

    app.router.add_get("/cluster-status", cluster_status)
    app.router.add_route("*", "/{tail:.*}", proxy)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Comma-separated listen addresses")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--backends", nargs="+", required=True)
    args = parser.parse_args()
    hosts = [item.strip() for item in args.host.split(",") if item.strip()]
    if not hosts:
        parser.error("--host must contain a listen address")
    web.run_app(create_app(args.backends), host=hosts if len(hosts) > 1 else hosts[0], port=args.port, access_log=None)


if __name__ == "__main__":
    main()
