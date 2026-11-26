"""Streaming responses that propagate HTTP disconnects to synthesis producers."""

from __future__ import annotations

from typing import Any

import anyio
from starlette.requests import ClientDisconnect
from starlette.responses import StreamingResponse


class ClosingStreamingResponse(StreamingResponse):
    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        except ClientDisconnect:
            pass
        finally:
            close = getattr(self.body_iterator, "aclose", None)
            if close is not None:
                with anyio.CancelScope(shield=True):
                    await close()
