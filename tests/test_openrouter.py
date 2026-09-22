"""HTTP engine: body build, event mapping, failure translation, lane behaviour.

Nothing here touches the network. `fake_openrouter` swaps the engine's shared
client for an httpx MockTransport, so the tests assert on the exact bytes the
engine would have sent and the exact CanonicalEvents it produces from a canned
stream — the same contract `test_engine.py` holds the CLI engine to.
"""
import asyncio
import json

import pytest

from conftest import OR_SUCCESS_LINES
from gateway import config, engine, lanes
from gateway.canonical import CanonicalMessage, CanonicalRequest, Delta, Error, Start, Stop
from gateway.engines import openrouter

FLASH = "openrouter/z-ai/glm-5.3-flash"
TEXT_ONLY = "openrouter/z-ai/glm-5.3"

PNG_B64 = "iVBORw0KGgo="


def _req(**kw):
    base = dict(model=FLASH, requested_model="glm-flash", system=None,
                messages=[CanonicalMessage("user", [{"type": "text", "text": "hi"}])],
                stream=True)
    base.update(kw)
    return CanonicalRequest(**base)


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "or-test-key")
    monkeypatch.setattr(config, "OPENROUTER_FALLBACK", True)
    lanes._semaphores.clear()
    yield
    lanes._semaphores.clear()


async def _drain(req):
    return [ev async for ev in openrouter.run_openrouter(req)]


# ---- body ---------------------------------------------------------------

def test_body_strips_our_routing_prefix():
    assert openrouter.build_body(_req())["model"] == "z-ai/glm-5.3-flash"


def test_body_puts_the_system_text_first_as_a_role():
    body = openrouter.build_body(_req(system="Be terse."))
    assert body["messages"][0] == {"role": "system", "content": "Be terse."}


def test_body_keeps_the_turns_as_turns():
    """Not the CLI's flattened transcript: this is a real chat API."""
    req = _req(messages=[
        CanonicalMessage("user", [{"type": "text", "text": "one"}]),
        CanonicalMessage("assistant", [{"type": "text", "text": "two"}]),
        CanonicalMessage("user", [{"type": "text", "text": "three"}]),
    ])
    assert [m["role"] for m in openrouter.build_body(req)["messages"]] == \
        ["user", "assistant", "user"]
    assert "[conversation so far]" not in json.dumps(openrouter.build_body(req))


def test_history_media_collapses_to_the_same_placeholder_as_the_cli():
    req = _req(messages=[
        CanonicalMessage("user", [{"type": "image", "media_type": "image/png", "data": PNG_B64}]),
        CanonicalMessage("user", [{"type": "text", "text": "and now?"}]),
    ])
    assert openrouter.build_body(req)["messages"][0]["content"] == "[image omitted]"


def test_a_final_image_is_sent_as_a_data_uri():
    req = _req(messages=[CanonicalMessage("user", [
        {"type": "text", "text": "what is this"},
        {"type": "image", "media_type": "image/png", "data": PNG_B64}])])
    parts = openrouter.build_body(req)["messages"][-1]["content"]
    assert parts[0] == {"type": "text", "text": "what is this"}
    assert parts[1]["image_url"]["url"] == f"data:image/png;base64,{PNG_B64}"


def test_sampling_knobs_are_honoured_here_even_though_the_cli_ignores_them():
    body = openrouter.build_body(_req(max_tokens=40, temperature=0.2, stop=["END"]))
    assert body["max_tokens"] == 40
    assert body["temperature"] == 0.2
    assert body["stop"] == ["END"]


def test_reasoning_is_off_when_the_model_map_disables_thinking():
    """The flash tier is mapped to max_thinking_tokens 0, and titling asks for 40
    tokens — with reasoning on, they would all be spent thinking."""
    assert openrouter.build_body(_req())["reasoning"] == {"enabled": False}


def test_an_effort_suffix_becomes_the_upstream_effort(monkeypatch):
    from gateway import models
    monkeypatch.setattr(models, "resolve_max_thinking_tokens", lambda m: None)
    assert openrouter.build_body(_req(effort_override="max"))["reasoning"] == {"effort": "high"}


def test_providers_are_sorted_by_latency_because_that_is_the_point():
    assert openrouter.build_body(_req())["provider"] == {"sort": "latency"}


async def test_the_key_rides_in_the_header_and_never_in_the_body(fake_openrouter):
    await _drain(_req())
    assert fake_openrouter["headers"]["authorization"] == "Bearer or-test-key"
    assert "or-test-key" not in json.dumps(fake_openrouter["body"])


# ---- event mapping ------------------------------------------------------

async def test_a_success_stream_becomes_one_start_two_deltas_and_a_stop(fake_openrouter):
    evs = await _drain(_req())
    assert [type(e) for e in evs] == [Start, Delta, Delta, Stop]
    assert [e.text for e in evs if isinstance(e, Delta)] == ["PI", "NG"]
    stop = evs[-1]
    assert (stop.stop_reason, stop.output_tokens, stop.input_tokens) == ("end_turn", 5, 42)


async def test_reasoning_text_is_never_yielded_as_the_answer(fake_openrouter):
    assert "hmm let me think" not in "".join(
        e.text for e in await _drain(_req()) if isinstance(e, Delta))


async def test_keepalive_comments_are_not_content(fake_openrouter):
    assert "OPENROUTER PROCESSING" not in "".join(
        e.text for e in await _drain(_req()) if isinstance(e, Delta))


async def test_length_becomes_max_tokens(fake_openrouter):
    fake_openrouter["lines"] = OR_SUCCESS_LINES[:-2] + [
        'data: {"choices":[{"delta":{},"finish_reason":"length"}],'
        '"usage":{"prompt_tokens":1,"completion_tokens":2}}\n',
        "data: [DONE]\n"]
    assert (await _drain(_req()))[-1].stop_reason == "max_tokens"


async def test_an_unknown_finish_reason_is_canonical_error(fake_openrouter):
    fake_openrouter["lines"] = OR_SUCCESS_LINES[:-2] + [
        'data: {"choices":[{"delta":{},"finish_reason":"content_filter"}]}\n',
        "data: [DONE]\n"]
    assert (await _drain(_req()))[-1].stop_reason == "error"


async def test_a_mid_stream_error_ends_the_turn_and_nothing_follows(fake_openrouter):
    fake_openrouter["lines"] = OR_SUCCESS_LINES[:4] + [
        'data: {"error":{"message":"upstream exploded","code":500}}\n',
        'data: {"choices":[{"delta":{"content":"never"}}]}\n']
    evs = await _drain(_req())
    assert [type(e) for e in evs] == [Start, Delta, Error]
    assert evs[-1].status == 502 and "exploded" in evs[-1].message


# ---- failure translation ------------------------------------------------

async def test_an_upstream_401_never_reaches_the_client_as_401(fake_openrouter):
    """It is OUR credential that is wrong. Forwarded, Nimbus would tell the person
    to sign in again over a key they have never seen."""
    fake_openrouter["status"] = 401
    with pytest.raises(openrouter.PreStreamFailure) as e:
        await _drain(_req())
    assert e.value.status == 502 and e.value.label == "auth"


@pytest.mark.parametrize("status,expect_status,label", [
    (429, 429, "429"),
    (402, 502, "no-credit"),
    (500, 502, "500"),
    (503, 503, "503"),
])
async def test_pre_stream_statuses_translate(fake_openrouter, status, expect_status, label):
    fake_openrouter["status"] = status
    with pytest.raises(openrouter.PreStreamFailure) as e:
        await _drain(_req())
    assert (e.value.status, e.value.label) == (expect_status, label)


async def test_a_stream_with_no_chunks_is_a_pre_stream_failure(fake_openrouter):
    fake_openrouter["lines"] = ["data: [DONE]\n"]
    with pytest.raises(openrouter.PreStreamFailure) as e:
        await _drain(_req())
    assert e.value.label == "no-output"


# ---- fallback -----------------------------------------------------------

async def test_a_pre_stream_failure_falls_back_to_claude(fake_openrouter, fake_claude):
    fake_openrouter["status"] = 429
    req = _req(requested_model="glm-flash")
    evs = [ev async for ev in engine.run(req)]
    assert [type(e) for e in evs] == [Start, Delta, Delta, Stop]
    assert req.engine == "cli"
    assert req.route_reason == "openrouter-429"
    assert req.model == "haiku"  # claude_fallback for the flash tier


async def test_fallback_can_be_switched_off(fake_openrouter, monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_FALLBACK", False)
    fake_openrouter["status"] = 429
    evs = [ev async for ev in engine.run(_req())]
    assert [type(e) for e in evs] == [Error] and evs[0].status == 429


async def test_a_failure_after_the_first_chunk_is_never_handed_over(fake_openrouter, fake_claude):
    """Half an answer is already on the wire; no other engine can take it."""
    fake_openrouter["lines"] = OR_SUCCESS_LINES[:4] + [
        'data: {"error":{"message":"died mid-stream"}}\n']
    req = _req()
    evs = [ev async for ev in engine.run(req)]
    assert isinstance(evs[-1], Error)
    assert req.engine == "openrouter"


# ---- lane ---------------------------------------------------------------

async def test_the_http_lane_saturates_without_touching_the_cli_lanes(fake_openrouter, monkeypatch):
    monkeypatch.setattr(config, "MAX_CONCURRENT_HTTP", 0)
    monkeypatch.setattr(config, "QUEUE_WAIT_MAX", 0.05)
    lanes._semaphores.clear()
    evs = await _drain(_req())
    assert [type(e) for e in evs] == [Error] and evs[0].status == 503
    assert "fast" not in lanes._semaphores and "heavy" not in lanes._semaphores


async def test_cancelling_mid_stream_frees_the_http_slot(fake_openrouter, monkeypatch):
    monkeypatch.setattr(config, "MAX_CONCURRENT_HTTP", 1)
    lanes._semaphores.clear()

    gen = openrouter.run_openrouter(_req())
    await gen.__anext__()                       # Start: the slot is held
    assert lanes.get_semaphore("http").locked()
    await gen.aclose()                          # the client went away
    assert not lanes.get_semaphore("http").locked()


# ---- usage accounting ---------------------------------------------------

async def test_the_record_carries_the_engine_the_provider_and_the_real_charge(
        fake_openrouter, monkeypatch):
    from gateway import usage_log
    recorded = []
    monkeypatch.setattr(usage_log, "record", lambda **kw: recorded.append(kw))
    # Through the dispatcher, because `engine` is written by select_engine — the
    # record has to show which engine actually answered, not which module ran.
    req = _req()
    assert [ev async for ev in engine.run(req)]
    rec = recorded[-1]
    assert rec["outcome"] == "success"
    assert rec["lane"] == "http"
    assert rec["provider"] == "Z.AI"
    assert rec["cost_usd"] == 0.000123
    assert rec["reasoning_tokens"] == 3
    assert rec["req"].engine == "openrouter"
