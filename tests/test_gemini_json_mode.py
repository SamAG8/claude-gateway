"""Gemini JSON mode: the instruction going out, and the fence coming back.

The adapter never read `generationConfig.responseMimeType`, so a client asking
for `application/json` got Claude's default chat-style output — a ```json fence,
sometimes followed by explanatory prose. Every downstream consumer's
fence-stripping regex was anchored to the end of the string (`\\s*```$`), so any
trailing prose broke `json.loads` — intermittently, only when the model chose to
add commentary, which is the worst way for a bug to present.

These tests are new. The fix shipped to production without any, in a downstream
vendored copy of this repository; this is the half that was missing.
"""
import json

import pytest

from conftest import TEST_KEY

pytestmark = pytest.mark.anyio

AUTH_G = {"x-goog-api-key": TEST_KEY}
JSON_MODE = {"responseMimeType": "application/json"}


def _reply(mock_engine, text: str) -> None:
    """Make the canned stream answer with exactly this text."""
    from gateway.canonical import Delta, Start, Stop

    mock_engine["events"] = [
        Start(model="claude-sonnet-4-6", input_tokens=11),
        Delta(text=text),
        Stop(stop_reason="end_turn", output_tokens=3, input_tokens=11),
    ]


async def _generate(client, mock_engine, text, gen=None):
    _reply(mock_engine, text)
    r = await client.post(
        "/v1beta/models/gemini-1.5-pro:generateContent",
        headers=AUTH_G,
        json={
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            **({"generationConfig": gen} if gen else {}),
        },
    )
    assert r.status_code == 200, r.text
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


# --- what goes OUT ------------------------------------------------------------


async def test_json_mode_tells_the_model_to_return_only_json(client, mock_engine):
    await _generate(client, mock_engine, "{}", gen=JSON_MODE)
    assert "ONLY the raw JSON" in (mock_engine["req"].system or "")


async def test_a_response_schema_also_means_json_mode(client, mock_engine):
    """A caller that supplies a schema has asked for JSON just as plainly as one
    that sets the mime type, and the SDKs let you do either."""
    await _generate(client, mock_engine, "{}", gen={"responseSchema": {"type": "object"}})
    assert "ONLY the raw JSON" in (mock_engine["req"].system or "")


async def test_the_instruction_is_appended_not_substituted(client, mock_engine):
    """A caller's own system instruction is not something to throw away."""
    _reply(mock_engine, "{}")
    r = await client.post(
        "/v1beta/models/gemini-1.5-pro:generateContent",
        headers=AUTH_G,
        json={
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "systemInstruction": {"parts": [{"text": "You are a surveyor."}]},
            "generationConfig": JSON_MODE,
        },
    )
    assert r.status_code == 200
    system = mock_engine["req"].system
    assert "You are a surveyor." in system
    assert "ONLY the raw JSON" in system


async def test_an_ordinary_request_is_left_alone(client, mock_engine):
    """No mime type, no schema: nothing is added, and nothing is stripped."""
    fenced = '```json\n{"a": 1}\n```'
    out = await _generate(client, mock_engine, fenced)
    assert out == fenced, "a non-JSON-mode response was rewritten"
    assert "ONLY the raw JSON" not in (mock_engine["req"].system or "")


# --- what comes BACK ----------------------------------------------------------


@pytest.mark.parametrize("wrapped,expected", [
    ('```json\n{"a": 1}\n```', '{"a": 1}'),
    ('```\n{"a": 1}\n```', '{"a": 1}'),
    # The one that actually broke production: prose AFTER the fence, which an
    # end-anchored regex could not strip.
    ('```json\n{"a": 1}\n```\n\nI hope that helps!', '{"a": 1}'),
    ('Here is the JSON you asked for:\n```json\n{"a": 1}\n```', '{"a": 1}'),
    # No fence at all, prose on both sides.
    ('Sure thing: {"a": 1} — let me know if you need more.', '{"a": 1}'),
    # A top-level array is JSON too.
    ('Here you go: [1, 2, 3]', '[1, 2, 3]'),
    # Already clean: unchanged.
    ('{"a": 1}', '{"a": 1}'),
])
async def test_the_json_value_is_extracted_however_it_was_wrapped(
    client, mock_engine, wrapped, expected,
):
    out = await _generate(client, mock_engine, wrapped, gen=JSON_MODE)
    assert out == expected
    json.loads(out)  # the point of all this: it parses


async def test_nesting_and_braces_inside_strings_survive(client, mock_engine):
    """The fallback matches braces, so a nested object must not be truncated at
    the first close — and a brace inside a string must not be counted."""
    value = '{"outer": {"inner": [1, 2]}, "note": "a } inside a string"}'
    out = await _generate(client, mock_engine, f"Here: {value}", gen=JSON_MODE)
    assert json.loads(out) == json.loads(value)


async def test_a_response_with_no_json_at_all_is_returned_unchanged(client, mock_engine):
    """Better to hand the caller what the model said than an empty string: they
    can see what went wrong. Extraction is best-effort, not a guarantee."""
    out = await _generate(client, mock_engine, "I could not answer that.", gen=JSON_MODE)
    assert out == "I could not answer that."


async def test_streaming_json_mode_still_gets_the_instruction(client, mock_engine):
    """A limit, stated rather than left implicit: the extraction is
    non-streaming only, because pulling a JSON value out of a stream means
    buffering the whole response — the one thing a streaming caller asked not to
    happen. The system instruction still applies, so the model is told to return
    raw JSON; it simply does not get the second line of defence.
    """
    _reply(mock_engine, '{"a": 1}')
    r = await client.post(
        "/v1beta/models/gemini-1.5-pro:streamGenerateContent?alt=sse",
        headers=AUTH_G,
        json={
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "generationConfig": JSON_MODE,
        },
    )
    assert r.status_code == 200
    assert "ONLY the raw JSON" in (mock_engine["req"].system or "")


async def test_a_backslash_before_a_quote_does_not_end_the_string(client, mock_engine):
    """The hardest case for a brace scanner: a JSON string whose last character
    is an escaped backslash, immediately before the closing quote. Treating that
    backslash as escaping the quote would leave the scanner inside a string
    forever and return the whole prose-wrapped blob."""
    BS = chr(92)
    value = '{"path": "C:' + BS + BS + 'dir' + BS + BS + '", "note": "} not structure"}'
    json.loads(value)  # the fixture itself is valid JSON

    out = await _generate(client, mock_engine, f"Here you go: {value} hope that helps", gen=JSON_MODE)
    assert json.loads(out) == json.loads(value)
