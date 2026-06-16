"""The deep renderer: drive the engine's CanonicalEvent stream once, for every protocol.

A Formatter is the seam — a small per-request, per-protocol adapter that only knows
how to render each event kind (and the non-streaming body). The drivers own the
ordering, termination, and the stream-vs-collect split that the three adapters used
to each re-implement.
"""
from typing import AsyncIterator, Iterable, Protocol

from fastapi.responses import JSONResponse, StreamingResponse

from . import engine
from .adapters._util import SSE_HEADERS
from .canonical import CanonicalRequest, Delta, Error, Result, Start, Stop


class Formatter(Protocol):
    # Streaming: each hook returns the SSE chunks to emit for that event (0..n).
    def on_start(self, ev: Start) -> Iterable[str]: ...
    def on_delta(self, ev: Delta) -> Iterable[str]: ...
    def on_stop(self, ev: Stop) -> Iterable[str]: ...
    def on_error(self, ev: Error) -> Iterable[str]: ...

    # Non-streaming: build the success body. Errors render via errors.py helpers.
    def complete(self, result: Result) -> dict: ...
    def error_response(self, status: int, message: str) -> JSONResponse: ...


async def _drive(req: CanonicalRequest, fmt: Formatter) -> AsyncIterator[str]:
    async for ev in engine.run_claude(req):
        if isinstance(ev, Start):
            for chunk in fmt.on_start(ev):
                yield chunk
        elif isinstance(ev, Delta):
            for chunk in fmt.on_delta(ev):
                yield chunk
        elif isinstance(ev, Stop):
            for chunk in fmt.on_stop(ev):
                yield chunk
            return
        elif isinstance(ev, Error):
            for chunk in fmt.on_error(ev):
                yield chunk
            return


def stream_response(req: CanonicalRequest, fmt: Formatter) -> StreamingResponse:
    return StreamingResponse(_drive(req, fmt), media_type="text/event-stream", headers=SSE_HEADERS)


async def complete_response(req: CanonicalRequest, fmt: Formatter) -> JSONResponse:
    result = await engine.collect(req)
    if result.error:
        return fmt.error_response(result.error.status, result.error.message)
    return JSONResponse(fmt.complete(result))


async def respond(req: CanonicalRequest, fmt: Formatter):
    """Single entry point: stream or complete based on the request."""
    if req.stream:
        return stream_response(req, fmt)
    return await complete_response(req, fmt)
