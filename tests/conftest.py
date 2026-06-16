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
