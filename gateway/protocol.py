"""The deep renderer: drive the engine's CanonicalEvent stream once, for every protocol.

A Formatter is the seam — a small per-request, per-protocol adapter that only knows
how to render each event kind (and the non-streaming body). The drivers own the
ordering, termination, and the stream-vs-collect split that the three adapters used
to each re-implement.
"""
import asyncio
from typing import AsyncIterator, Iterable, Optional, Protocol

from fastapi.responses import JSONResponse, StreamingResponse
from starlette.requests import Request

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
    started = False
    async for ev in engine.run_claude(req):
        if isinstance(ev, Start):
            # Defense in depth: one HTTP stream represents one public model
            # message even when an upstream engine performs internal tool turns.
            # Never let a duplicate Start escape as an invalid protocol sequence.
            if started:
                continue
            started = True
            for chunk in fmt.on_start(ev):
                yield chunk
        elif isinstance(ev, Delta):
            if not started:
                continue
            for chunk in fmt.on_delta(ev):
                yield chunk
        elif isinstance(ev, Stop):
            if not started:
                for chunk in fmt.on_error(Error(502, "upstream ended before message_start")):
                    yield chunk
                return
            for chunk in fmt.on_stop(ev):
                yield chunk
            return
        elif isinstance(ev, Error):
            for chunk in fmt.on_error(ev):
                yield chunk
            return


def stream_response(req: CanonicalRequest, fmt: Formatter) -> StreamingResponse:
    return StreamingResponse(_drive(req, fmt), media_type="text/event-stream", headers=SSE_HEADERS)


async def _collect_with_disconnect(req: CanonicalRequest, request: Request) -> Result:
    """Run engine.collect as a cancellable task and race it against the client
    disconnecting. On disconnect we cancel the task, which propagates into
    engine.run_claude's `except asyncio.CancelledError` (A3) — that kills the CLI
    subprocess and frees its lane slot immediately instead of holding it for the
    full gateway TIMEOUT while the abandoned client has already moved on."""
    task = asyncio.ensure_future(engine.collect(req))
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=0.25)
            if task in done:
                return task.result()
            if await request.is_disconnected():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                # Client is gone; the return value is never rendered. Surface a
                # 499-style error so the caller path stays consistent.
                return Result(text="", model=req.requested_model, stop_reason="end_turn",
                              input_tokens=0, output_tokens=0,
                              error=Error(499, "client disconnected"))
    except asyncio.CancelledError:
        # Our own driver was cancelled (e.g. server shutdown): cancel the child too.
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        raise


async def complete_response(req: CanonicalRequest, fmt: Formatter,
                            request: Optional[Request] = None) -> JSONResponse:
    if request is not None:
        result = await _collect_with_disconnect(req, request)
    else:
        result = await engine.collect(req)
    if result.error:
        return fmt.error_response(result.error.status, result.error.message)
    return JSONResponse(fmt.complete(result))


async def respond(req: CanonicalRequest, fmt: Formatter,
                  request: Optional[Request] = None):
    """Single entry point: stream or complete based on the request.

    ``request`` (the Starlette Request) is optional for back-compat; when supplied,
    the non-streaming path cancels the underlying CLI run if the client disconnects
    (A3). The streaming path already gets cancellation for free: Starlette cancels
    the StreamingResponse generator on disconnect, which propagates CancelledError
    into run_claude and kills the subprocess."""
    if req.stream:
        return stream_response(req, fmt)
    return await complete_response(req, fmt, request)
