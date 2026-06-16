"""Shared test fixtures.

`fake_claude` monkeypatches asyncio.create_subprocess_exec so engine tests can
drive run_claude with a canned stream-json transcript instead of the real CLI.
"""
import asyncio
import json

import pytest


class _FakeStdin:
    def __init__(self):
        self.written = b""

    def write(self, data):
        self.written += data

    async def drain(self):
        pass

    def close(self):
        pass


class _FakeStream:
    def __init__(self, lines: list[bytes]):
        self._lines = list(lines)

    async def readline(self) -> bytes:
        return self._lines.pop(0) if self._lines else b""

    async def read(self) -> bytes:
        out = b"".join(self._lines)
        self._lines = []
        return out


class FakeProcess:
    def __init__(self, stdout_lines: list[bytes], stderr_lines: list[bytes] | None = None):
        self.stdin = _FakeStdin()
        self.stdout = _FakeStream(stdout_lines)
        self.stderr = _FakeStream(stderr_lines or [])
        self.killed = False
        self.returncode = 0

    def kill(self):
        self.killed = True

    async def wait(self):
        return self.returncode


def _line(obj: dict) -> bytes:
    return (json.dumps(obj) + "\n").encode()


# A realistic success transcript (mirrors the live CLI: message_start ->
# content_block_delta(s) -> message_delta -> result), with noise lines the
# engine must ignore.
SUCCESS_LINES = [
    _line({"type": "system", "subtype": "init", "model": "claude-sonnet-4-6"}),
    _line({"type": "stream_event", "event": {
        "type": "message_start",
        "message": {"model": "claude-sonnet-4-6", "id": "msg_1", "role": "assistant",
                    "usage": {"input_tokens": 136, "output_tokens": 1}}}}),
    _line({"type": "stream_event", "event": {
        "type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}}),
    _line({"type": "stream_event", "event": {
        "type": "content_block_delta", "index": 0,
        "delta": {"type": "text_delta", "text": "PI"}}}),
    _line({"type": "stream_event", "event": {
        "type": "content_block_delta", "index": 0,
        "delta": {"type": "text_delta", "text": "NG"}}}),
    _line({"type": "assistant", "message": {"content": [{"type": "text", "text": "PING"}]}}),
    _line({"type": "stream_event", "event": {"type": "content_block_stop", "index": 0}}),
    _line({"type": "stream_event", "event": {
        "type": "message_delta", "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 5}}}),
    _line({"type": "stream_event", "event": {"type": "message_stop"}}),
    _line({"type": "rate_limit_event", "rate_limit_info": {"status": "allowed"}}),
    _line({"type": "result", "subtype": "success", "is_error": False, "result": "PING",
           "stop_reason": "end_turn", "usage": {"input_tokens": 136, "output_tokens": 5}}),
]

ERROR_LINES = [
    _line({"type": "result", "subtype": "error_during_execution", "is_error": True,
           "result": "boom"}),
]


@pytest.fixture
def fake_claude(monkeypatch):
    """Patch create_subprocess_exec to return a FakeProcess; yields a holder.

    Set ``holder['lines']`` before triggering run_claude to choose the transcript.
    ``holder['proc']`` exposes the spawned FakeProcess (e.g. to read stdin).
    """
    holder = {"lines": SUCCESS_LINES, "stderr": [], "proc": None, "argv": None}

    async def fake_exec(*args, **kwargs):
        holder["argv"] = list(args)
        proc = FakeProcess(holder["lines"], holder["stderr"])
        holder["proc"] = proc
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    return holder


# --- Adapter-level fixtures (mock the engine, not the subprocess) ---------

TEST_KEY = "testkey"


@pytest.fixture
def mock_engine(monkeypatch):
    """Patch engine.run_claude with a canned canonical stream and capture the request.

    ``state['req']`` holds the CanonicalRequest the adapter produced; set
    ``state['events']`` to override the yielded canonical events.
    """
    from gateway import config, engine

    state = {
        "req": None,
        "events": [
            {"t": "start", "model": "claude-sonnet-4-6", "input_tokens": 11},
            {"t": "delta", "text": "Hello"},
            {"t": "delta", "text": " there"},
            {"t": "stop", "stop_reason": "end_turn", "output_tokens": 3, "input_tokens": 11},
        ],
    }

    async def fake_run(req):
        state["req"] = req
        for ev in state["events"]:
            yield ev

    monkeypatch.setattr(engine, "run_claude", fake_run)
    monkeypatch.setattr(config, "API_KEYS", {TEST_KEY})
    return state


@pytest.fixture
async def client():
    from httpx import ASGITransport, AsyncClient

    from main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def sse_events(text: str) -> list[str]:
    """Split an SSE body into the raw payload of each `data:` line."""
    out = []
    for block in text.strip().split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data:"):
                out.append(line[len("data:"):].strip())
    return out
