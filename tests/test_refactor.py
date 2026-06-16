"""Unit tests for the deepened seams: error taxonomy, translation helpers, renderer."""
import json

import pytest

from gateway import engine, errors, protocol
from gateway.canonical import Delta, Error, Start, Stop, map_reason
from gateway.translate import join_texts, to_role


# --- error taxonomy: one status table drives three protocols --------------

def _body(resp):
    return json.loads(resp.body)


def test_413_renders_per_protocol():
    assert _body(errors.anthropic_error(413, "x"))["error"]["type"] == "request_too_large"
    assert _body(errors.openai_error(413, "x"))["error"]["type"] == "invalid_request_error"
    assert _body(errors.gemini_error(413, "x"))["error"]["status"] == "INVALID_ARGUMENT"


def test_500_vs_502_distinction_preserved():
    # server (500) and unavailable (502/504) stay distinct where the protocol cares.
    assert _body(errors.gemini_error(500, "x"))["error"]["status"] == "INTERNAL"
    assert _body(errors.gemini_error(502, "x"))["error"]["status"] == "UNAVAILABLE"
    assert _body(errors.gemini_error(504, "x"))["error"]["status"] == "UNAVAILABLE"
    # Anthropic/OpenAI collapse both to api_error.
    assert _body(errors.anthropic_error(502, "x"))["error"]["type"] == "api_error"
    assert _body(errors.openai_error(504, "x"))["error"]["type"] == "api_error"


def test_401_auth_kind():
    assert _body(errors.anthropic_error(401, "x"))["error"]["type"] == "authentication_error"
    assert _body(errors.gemini_error(401, "x"))["error"]["status"] == "UNAUTHENTICATED"


# --- shared translation helpers ------------------------------------------

def test_join_texts_filters_and_joins():
    assert join_texts([{"text": "a"}, {"text": ""}, {"nope": 1}, {"text": "b"}]) == "a\nb"
    assert join_texts([]) is None
    assert join_texts([{"text": ""}]) is None
    assert join_texts([{"text": "a"}, {"text": "b"}], sep=" ") == "a b"


@pytest.mark.parametrize("role,aliases,expected", [
    ("assistant", ("assistant",), "assistant"),
    ("user", ("assistant",), "user"),
    ("model", ("model", "assistant"), "assistant"),
    ("system", ("assistant",), "user"),
    (None, ("assistant",), "user"),
])
def test_to_role(role, aliases, expected):
    assert to_role(role, aliases) == expected


def test_map_reason():
    assert map_reason({"end_turn": "STOP"}, "end_turn", "OTHER") == "STOP"
    assert map_reason({"end_turn": "STOP"}, "error", "OTHER") == "OTHER"


# --- the renderer drives a formatter and owns termination -----------------

class _FakeFmt:
    def __init__(self):
        self.calls = []

    def on_start(self, ev):
        self.calls.append("start")
        return ["S"]

    def on_delta(self, ev):
        self.calls.append("delta")
        return [f"D:{ev.text}"]

    def on_stop(self, ev):
        self.calls.append("stop")
        return ["P"]

    def on_error(self, ev):
        self.calls.append("error")
        return ["X"]


def _patch_stream(monkeypatch, events):
    async def fake_run(req):
        for e in events:
            yield e
    monkeypatch.setattr(engine, "run_claude", fake_run)


async def test_drive_renders_in_order(monkeypatch):
    _patch_stream(monkeypatch, [Start("m", 5), Delta("a"), Delta("b"), Stop("end_turn", 2, 5)])
    fmt = _FakeFmt()
    out = [c async for c in protocol._drive(None, fmt)]
    assert out == ["S", "D:a", "D:b", "P"]
    assert fmt.calls == ["start", "delta", "delta", "stop"]


async def test_drive_terminates_on_stop(monkeypatch):
    # anything after Stop must not be rendered
    _patch_stream(monkeypatch, [Start("m", 5), Stop("end_turn", 1, 5), Delta("late")])
    fmt = _FakeFmt()
    out = [c async for c in protocol._drive(None, fmt)]
    assert out == ["S", "P"]
    assert "delta" not in fmt.calls


async def test_drive_terminates_on_error(monkeypatch):
    _patch_stream(monkeypatch, [Error(502, "boom"), Delta("after")])
    fmt = _FakeFmt()
    out = [c async for c in protocol._drive(None, fmt)]
    assert out == ["X"]
    assert fmt.calls == ["error"]
