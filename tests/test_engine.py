"""Engine tests: argv build, stdin build, stream parsing, error/timeout handling."""
import json

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

from conftest import ERROR_LINES


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
