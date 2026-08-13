"""Engine tests: argv build, stdin build, stream parsing, error/timeout handling."""
import json
import sys

import pytest

from gateway import config, engine
from gateway.canonical import (
    CanonicalMessage,
    CanonicalRequest,
    Delta,
    Error,
    Start,
    Stop,
    map_stop_reason,
)

from conftest import ERROR_LINES, SUCCESS_LINES, _line


def _req(**kw):
    base = dict(model="sonnet", requested_model="gpt-4o", system=None,
                messages=[CanonicalMessage("user", [{"type": "text", "text": "hi"}])],
                stream=True)
    base.update(kw)
    return CanonicalRequest(**base)


async def _drain(req):
    return [ev async for ev in engine.run_claude(req)]


# ---- argv ---------------------------------------------------------------

def test_build_argv_disables_tools_and_settings():
    argv = engine.build_argv(_req())
    # --tools must be the empty string (the old `--tools none` bug), and present.
    ti = argv.index("--tools")
    assert argv[ti + 1] == ""
    si = argv.index("--setting-sources")
    assert argv[si + 1] == ""
    assert "--no-session-persistence" in argv
    assert "--include-partial-messages" in argv
    mi = argv.index("--model")
    assert argv[mi + 1] == "sonnet"


def test_build_argv_default_system_prompt():
    argv = engine.build_argv(_req(system=None))
    sp = argv.index("--system-prompt")
    assert argv[sp + 1] == config.DEFAULT_SYSTEM_PROMPT


def test_build_argv_custom_system_prompt():
    argv = engine.build_argv(_req(system="Be terse."))
    sp = argv.index("--system-prompt")
    assert argv[sp + 1] == "Be terse."


def test_build_argv_haiku_uses_low_effort(monkeypatch):
    # Fast tier stays at low effort even when the global EFFORT is high (issue #11).
    monkeypatch.setattr(config, "EFFORT", "high")
    argv = engine.build_argv(_req(model="haiku"))
    assert argv[argv.index("--effort") + 1] == "low"


def test_build_argv_non_haiku_uses_global_effort(monkeypatch):
    monkeypatch.setattr(config, "EFFORT", "high")
    argv = engine.build_argv(_req(model="opus"))
    assert argv[argv.index("--effort") + 1] == "high"


def test_build_argv_no_effort_when_global_unset(monkeypatch):
    monkeypatch.setattr(config, "EFFORT", "")
    assert "--effort" not in engine.build_argv(_req(model="opus"))


# ---- stdin --------------------------------------------------------------

def test_build_stdin_single_turn_preserves_image_blocks():
    req = _req(messages=[CanonicalMessage("user", [
        {"type": "text", "text": "what is this?"},
        {"type": "image", "media_type": "image/png", "data": "AAAA"},
    ])])
    msg = json.loads(engine.build_stdin(req))
    content = msg["message"]["content"]
    assert content[0] == {"type": "text", "text": "what is this?"}
    assert content[1]["type"] == "image"
    assert content[1]["source"] == {"type": "base64", "media_type": "image/png", "data": "AAAA"}


def test_build_stdin_single_turn_preserves_document_blocks():
    req = _req(messages=[CanonicalMessage("user", [
        {"type": "text", "text": "extract this"},
        {"type": "document", "media_type": "application/pdf", "data": "BBBB"},
    ])])
    msg = json.loads(engine.build_stdin(req))
    content = msg["message"]["content"]
    assert content[0] == {"type": "text", "text": "extract this"}
    assert content[1]["type"] == "document"
    assert content[1]["source"] == {"type": "base64", "media_type": "application/pdf", "data": "BBBB"}


def test_build_stdin_multiturn_flattens_history_and_keeps_final_image():
    req = _req(messages=[
        CanonicalMessage("user", [{"type": "text", "text": "first"}]),
        CanonicalMessage("assistant", [{"type": "text", "text": "ok"}]),
        CanonicalMessage("user", [
            {"type": "text", "text": "second"},
            {"type": "image", "media_type": "image/png", "data": "ZZZ"},
        ]),
    ])
    msg = json.loads(engine.build_stdin(req))
    content = msg["message"]["content"]
    text = content[0]["text"]
    assert "User: first" in text and "Assistant: ok" in text
    assert text.rstrip().endswith("second")
    # final-turn image preserved as a real image block
    assert content[1]["type"] == "image" and content[1]["source"]["data"] == "ZZZ"


def test_build_stdin_history_image_becomes_placeholder():
    req = _req(messages=[
        CanonicalMessage("user", [{"type": "image", "media_type": "image/png", "data": "X"}]),
        CanonicalMessage("assistant", [{"type": "text", "text": "ok"}]),
        CanonicalMessage("user", [{"type": "text", "text": "now"}]),
    ])
    msg = json.loads(engine.build_stdin(req))
    assert "[image omitted]" in msg["message"]["content"][0]["text"]


def test_build_stdin_history_document_becomes_placeholder():
    req = _req(messages=[
        CanonicalMessage("user", [{"type": "document", "media_type": "application/pdf", "data": "X"}]),
        CanonicalMessage("assistant", [{"type": "text", "text": "ok"}]),
        CanonicalMessage("user", [{"type": "text", "text": "now"}]),
    ])
    msg = json.loads(engine.build_stdin(req))
    assert "[document omitted]" in msg["message"]["content"][0]["text"]


# ---- stop reason mapping ------------------------------------------------

@pytest.mark.parametrize("cli,expected", [
    ("end_turn", "end_turn"),
    ("max_tokens", "max_tokens"),
    ("stop_sequence", "error"),
    (None, "error"),
])
def test_map_stop_reason(cli, expected):
    assert map_stop_reason(cli) == expected


def test_map_stop_reason_is_error_overrides():
    assert map_stop_reason("end_turn", is_error=True) == "error"


# ---- stream parsing -----------------------------------------------------

async def test_run_claude_yields_canonical_events(fake_claude):
    events = await _drain(_req())
    assert events[0] == Start(model="claude-sonnet-4-6", input_tokens=136)
    deltas = [e.text for e in events if isinstance(e, Delta)]
    assert "".join(deltas) == "PING"
    assert events[-1] == Stop(stop_reason="end_turn", output_tokens=5, input_tokens=136)


async def test_mcp_internal_turns_form_one_public_message(fake_claude):
    """MCP tool loops may start several internal Claude messages. The gateway
    must expose one public message lifecycle so Anthropic SDKs can consume it."""
    from conftest import _line

    fake_claude["lines"] = [
        _line({"type": "stream_event", "event": {
            "type": "message_start", "message": {
                "model": "claude-sonnet-4-6", "usage": {"input_tokens": 10}}}}),
        _line({"type": "stream_event", "event": {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": "Checking. "}}}),
        _line({"type": "stream_event", "event": {"type": "message_stop"}}),
        _line({"type": "stream_event", "event": {
            "type": "message_start", "message": {
                "model": "claude-sonnet-4-6", "usage": {"input_tokens": 25}}}}),
        _line({"type": "stream_event", "event": {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": "Final answer."}}}),
        _line({"type": "stream_event", "event": {
            "type": "message_delta", "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 8}}}),
        _line({"type": "stream_event", "event": {"type": "message_stop"}}),
        _line({"type": "result", "subtype": "success", "is_error": False,
               "result": "Checking. Final answer.", "stop_reason": "end_turn",
               "usage": {"input_tokens": 25, "output_tokens": 8}}),
    ]

    events = await _drain(_req())
    assert len([e for e in events if isinstance(e, Start)]) == 1
    assert "".join(e.text for e in events if isinstance(e, Delta)) == "Checking. Final answer."
    assert events[-1] == Stop(stop_reason="end_turn", output_tokens=8, input_tokens=25)


async def test_collect_assembles_single_result(fake_claude):
    out = await engine.collect(_req(stream=False))
    assert out.text == "PING"
    assert out.model == "claude-sonnet-4-6"
    assert out.stop_reason == "end_turn"
    assert out.input_tokens == 136 and out.output_tokens == 5
    assert out.error is None


async def test_run_claude_surfaces_cli_error(fake_claude):
    fake_claude["lines"] = ERROR_LINES
    events = await _drain(_req())
    assert isinstance(events[-1], Error)
    assert events[-1].status == 502
    assert "boom" in events[-1].message


async def test_run_claude_stdin_is_written(fake_claude):
    await _drain(_req())
    written = json.loads(fake_claude["proc"].stdin.written)
    assert written["type"] == "user"
    assert written["message"]["content"][0]["text"] == "hi"


async def test_bare_mode_requires_anthropic_key(fake_claude, monkeypatch):
    monkeypatch.setattr(config, "ISOLATION_MODE", "bare")
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    events = await _drain(_req())
    assert events == [Error(500,
                            "ISOLATION_MODE=bare requires ANTHROPIC_API_KEY in the environment")]


# ---- stream line limit (issue #11) --------------------------------------

async def test_run_claude_survives_stream_lines_over_64kib(tmp_path, monkeypatch):
    """A >64 KiB NDJSON line (the CLI echoing an inline image) must not error.

    Uses a REAL subprocess so the real asyncio StreamReader limit is exercised —
    the fake_claude fixture bypasses it entirely. Pre-fix this fails with
    Error(500, 'Separator is found, but chunk is longer than limit').
    """
    big_echo = _line({"type": "user", "message": {
        "role": "user", "content": [{"type": "text", "text": "x" * 300_000}]}})
    transcript = tmp_path / "transcript.ndjson"
    transcript.write_bytes(SUCCESS_LINES[0] + big_echo + b"".join(SUCCESS_LINES[1:]))
    feeder = (
        "import sys\n"
        "sys.stdin.buffer.read()\n"  # drain the gateway's stdin write
        f"sys.stdout.buffer.write(open({str(transcript)!r}, 'rb').read())\n"
    )
    monkeypatch.setattr(engine, "build_argv", lambda req: [sys.executable, "-c", feeder])
    events = await _drain(_req())
    assert not any(isinstance(e, Error) for e in events)
    assert events[0] == Start(model="claude-sonnet-4-6", input_tokens=136)
    assert events[-1] == Stop(stop_reason="end_turn", output_tokens=5, input_tokens=136)


async def test_spawn_sets_stream_limit(fake_claude):
    await _drain(_req())
    assert fake_claude["kwargs"]["limit"] >= 32 * 1024 * 1024


async def test_spawn_limit_scales_with_stdin(fake_claude, monkeypatch):
    monkeypatch.setattr(config, "STREAM_LIMIT", 1024)
    req = _req(messages=[CanonicalMessage("user", [
        {"type": "image", "media_type": "image/jpeg", "data": "A" * 100_000}])])
    await _drain(req)
    assert fake_claude["kwargs"]["limit"] == 2 * len(engine.build_stdin(req))


# ---- A2: two-lane semaphore + bounded queue wait ------------------------

def _reset_semaphores():
    engine._semaphores.clear()


async def test_lane_selection_fast_vs_heavy(fake_claude, monkeypatch):
    """haiku picks the fast lane, sonnet the heavy lane (logged + usage-recorded)."""
    _reset_semaphores()
    recorded = []
    monkeypatch.setattr(engine.usage_log, "record",
                        lambda **kw: recorded.append(kw))
    await _drain(_req(model="haiku"))
    await _drain(_req(model="sonnet"))
    lanes = [r["lane"] for r in recorded]
    assert lanes == ["fast", "heavy"]
    assert all(r["queue_wait_ms"] is not None for r in recorded)
    assert all(r["first_event_ms"] is not None for r in recorded)
    assert all(r["first_text_ms"] is not None for r in recorded)
    assert all(r["total_ms"] is not None for r in recorded)
    assert all(r["prompt_bytes"] > 0 for r in recorded)
    assert all(r["mcp"] is False for r in recorded)


async def test_exact_haiku_id_uses_fast_lane_and_disables_thinking(fake_claude, monkeypatch):
    """Nimbus sends dated Claude ids, not the short ``haiku`` alias."""
    _reset_semaphores()
    recorded = []
    monkeypatch.setattr(engine.usage_log, "record", lambda **kw: recorded.append(kw))
    await _drain(_req(model="claude-haiku-4-5-20251001"))
    assert recorded[0]["lane"] == "fast"
    assert fake_claude["kwargs"]["env"]["MAX_THINKING_TOKENS"] == "0"


async def test_saturated_returns_503_and_releases(monkeypatch):
    """When a lane is full past QUEUE_WAIT_MAX, the request fails fast with 503
    and does NOT leak a slot (the held slot is released, so a later call succeeds)."""
    _reset_semaphores()
    monkeypatch.setattr(config, "QUEUE_WAIT_MAX", 0.05)
    # Force the heavy lane to capacity 1 and pre-acquire it.
    monkeypatch.setattr(config, "MAX_CONCURRENT", 1)
    sem = engine._get_semaphore("heavy")
    await sem.acquire()  # occupy the only slot
    events = await _drain(_req(model="sonnet"))
    assert isinstance(events[-1], Error) and events[-1].status == 503
    assert "saturated" in events[-1].message
    # We never entered the run body, so nothing extra was released; free our slot.
    sem.release()
    assert sem._value == 1  # back to full capacity, no double-release


async def test_slot_released_on_success(fake_claude):
    """A normal run releases its slot on completion (semaphore returns to full)."""
    _reset_semaphores()
    await _drain(_req(model="haiku"))
    sem = engine._get_semaphore("fast")
    assert sem._value == config.MAX_CONCURRENT_FAST


async def test_two_users_run_concurrently_with_isolated_mcp_tokens(monkeypatch):
    """Two Nimbus users get separate subprocess argv/stdin/streams and may use
    heavy-lane capacity concurrently; neither user's MCP identity is reused."""
    import asyncio as _asyncio
    from conftest import FakeProcess, SUCCESS_LINES

    _reset_semaphores()
    monkeypatch.setattr(config, "MAX_CONCURRENT", 2)
    monkeypatch.setattr(config, "MCP_SERVER_URL", "https://mcp.test")
    spawned = []
    both_spawned = _asyncio.Event()

    class _BarrierStream:
        def __init__(self):
            self.lines = list(SUCCESS_LINES)

        async def readline(self):
            await _asyncio.wait_for(both_spawned.wait(), timeout=1)
            return self.lines.pop(0) if self.lines else b""

        async def read(self):
            return b""

    async def fake_exec(*argv, **kwargs):
        proc = FakeProcess([])
        proc.stdout = _BarrierStream()
        spawned.append({"argv": list(argv), "kwargs": kwargs, "proc": proc})
        if len(spawned) == 2:
            both_spawned.set()
        return proc

    monkeypatch.setattr(_asyncio, "create_subprocess_exec", fake_exec)
    req_a = _req(model="sonnet", mcp_token="user-a-token")
    req_b = _req(model="sonnet", mcp_token="user-b-token")
    events_a, events_b = await _asyncio.gather(_drain(req_a), _drain(req_b))

    assert len(spawned) == 2  # both reached subprocess execution before either completed
    configs = []
    for call in spawned:
        argv = call["argv"]
        configs.append(json.loads(argv[argv.index("--mcp-config") + 1]))
        assert call["proc"].stdin.written  # each process received its own request body
    auth_headers = {
        cfg["mcpServers"][config.MCP_SERVER_NAME]["headers"]["Authorization"]
        for cfg in configs
    }
    assert auth_headers == {"Bearer user-a-token", "Bearer user-b-token"}
    assert events_a[-1].stop_reason == events_b[-1].stop_reason == "end_turn"


# ---- A3: kill-on-cancel frees the slot ----------------------------------

async def test_cancel_kills_subprocess_and_frees_slot(monkeypatch):
    """Cancelling a run mid-stream kills the CLI subprocess and releases the slot."""
    import asyncio as _asyncio
    _reset_semaphores()

    class _HangingStream:
        async def readline(self):
            await _asyncio.sleep(3600)  # never yields a line
        async def read(self):
            return b""

    from conftest import FakeProcess
    proc = FakeProcess([])
    proc.stdout = _HangingStream()

    async def fake_exec(*a, **k):
        return proc
    monkeypatch.setattr(_asyncio, "create_subprocess_exec", fake_exec)

    async def run():
        async for _ in engine.run_claude(_req(model="haiku")):
            pass

    task = _asyncio.ensure_future(run())
    await _asyncio.sleep(0.05)  # let it spawn + block on readline
    task.cancel()
    with pytest.raises(_asyncio.CancelledError):
        await task
    assert proc.killed is True
    sem = engine._get_semaphore("fast")
    assert sem._value == config.MAX_CONCURRENT_FAST  # slot freed
