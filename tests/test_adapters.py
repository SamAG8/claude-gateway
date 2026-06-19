"""Adapter shape/auth/image tests with a mocked engine (no real CLI)."""
import base64
import json

import pytest

from conftest import TEST_KEY, sse_events
from gateway.canonical import Error

# 1x1 transparent PNG
PNG_B64 = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQ"
           "AAAABJRU5ErkJggg==")

# Minimal valid base64 (the adapter only validates/decodes; it does not parse the PDF)
PDF_B64 = "JVBERi0xLjQK"  # "%PDF-1.4\n"

AUTH_A = {"x-api-key": TEST_KEY}
AUTH_O = {"Authorization": f"Bearer {TEST_KEY}"}
AUTH_G = {"x-goog-api-key": TEST_KEY}


# ====================== Anthropic ======================

async def test_anthropic_nonstream_shape(client, mock_engine):
    r = await client.post("/v1/messages", headers=AUTH_A, json={
        "model": "claude-sonnet-4-6", "max_tokens": 50,
        "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "message" and body["role"] == "assistant"
    assert body["model"] == "claude-sonnet-4-6"
    assert body["content"] == [{"type": "text", "text": "Hello there"}]
    assert body["stop_reason"] == "end_turn"
    assert body["usage"] == {"input_tokens": 11, "output_tokens": 3}


async def test_anthropic_stream_sequence(client, mock_engine):
    r = await client.post("/v1/messages", headers=AUTH_A, json={
        "model": "claude-sonnet-4-6", "max_tokens": 50, "stream": True,
        "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    types = [json.loads(d)["type"] for d in sse_events(r.text)]
    assert types == ["message_start", "content_block_start", "ping",
                     "content_block_delta", "content_block_delta",
                     "content_block_stop", "message_delta", "message_stop"]
    payloads = [json.loads(d) for d in sse_events(r.text)]
    assert payloads[0]["message"]["usage"]["input_tokens"] == 11
    text = "".join(p["delta"]["text"] for p in payloads if p["type"] == "content_block_delta")
    assert text == "Hello there"
    assert payloads[-2]["delta"]["stop_reason"] == "end_turn"


async def test_anthropic_auth_401(client, mock_engine):
    r = await client.post("/v1/messages", json={"model": "x", "messages": [
        {"role": "user", "content": "hi"}]})
    assert r.status_code == 401
    assert r.json()["error"]["type"] == "authentication_error"


async def test_anthropic_image_reaches_engine(client, mock_engine):
    await client.post("/v1/messages", headers=AUTH_A, json={
        "model": "claude-sonnet-4-6", "max_tokens": 50, "messages": [{
            "role": "user", "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image", "source": {"type": "base64",
                                             "media_type": "image/png", "data": PNG_B64}}]}]})
    blocks = mock_engine["req"].messages[-1].blocks
    img = [b for b in blocks if b["type"] == "image"]
    assert img and img[0]["media_type"] == "image/png" and img[0]["data"] == PNG_B64


# ====================== OpenAI ======================

async def test_openai_nonstream_shape(client, mock_engine):
    r = await client.post("/v1/chat/completions", headers=AUTH_O, json={
        "model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "gpt-4o"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "Hello there"}
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}


async def test_openai_stream_sequence_and_done(client, mock_engine):
    r = await client.post("/v1/chat/completions", headers=AUTH_O, json={
        "model": "gpt-4o", "stream": True, "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": "hi"}]})
    raw = sse_events(r.text)
    assert raw[-1] == "[DONE]"
    chunks = [json.loads(d) for d in raw if d != "[DONE]"]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks
                    if c["choices"])
    assert text == "Hello there"
    # finish chunk then usage chunk
    assert any(c["choices"] and c["choices"][0]["finish_reason"] == "stop" for c in chunks)
    usage = [c for c in chunks if c.get("usage")]
    assert usage and usage[0]["usage"]["total_tokens"] == 14


async def test_openai_system_message_merged(client, mock_engine):
    await client.post("/v1/chat/completions", headers=AUTH_O, json={
        "model": "gpt-4o", "messages": [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "hi"}]})
    assert mock_engine["req"].system == "Be terse."
    assert mock_engine["req"].messages[0].role == "user"


async def test_openai_auth_401(client, mock_engine):
    r = await client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": [
        {"role": "user", "content": "hi"}]})
    assert r.status_code == 401
    assert r.json()["error"]["type"] == "authentication_error"


async def test_openai_image_data_uri_reaches_engine(client, mock_engine):
    await client.post("/v1/chat/completions", headers=AUTH_O, json={
        "model": "gpt-4o", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{PNG_B64}"}}]}]})
    blocks = mock_engine["req"].messages[-1].blocks
    img = [b for b in blocks if b["type"] == "image"]
    assert img and img[0]["media_type"] == "image/png" and img[0]["data"] == PNG_B64


async def test_openai_non_data_url_rejected(client, mock_engine):
    r = await client.post("/v1/chat/completions", headers=AUTH_O, json={
        "model": "gpt-4o", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}]}]})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"


async def test_openai_models_list(client, mock_engine):
    r = await client.get("/v1/models", headers=AUTH_O)
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    ids = [m["id"] for m in body["data"]]
    assert "gpt-4o" in ids and "sonnet" in ids


# ====================== Gemini ======================

async def test_gemini_nonstream_shape(client, mock_engine):
    r = await client.post("/v1beta/models/gemini-1.5-pro:generateContent", headers=AUTH_G,
                          json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]})
    assert r.status_code == 200
    body = r.json()
    cand = body["candidates"][0]
    assert cand["content"] == {"role": "model", "parts": [{"text": "Hello there"}]}
    assert cand["finishReason"] == "STOP"
    assert body["usageMetadata"] == {"promptTokenCount": 11, "candidatesTokenCount": 3,
                                     "totalTokenCount": 14}


async def test_gemini_stream_partials(client, mock_engine):
    r = await client.post("/v1beta/models/gemini-1.5-pro:streamGenerateContent?alt=sse",
                          headers=AUTH_G,
                          json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]})
    payloads = [json.loads(d) for d in sse_events(r.text)]
    text = "".join(p["candidates"][0]["content"]["parts"][0]["text"] for p in payloads)
    assert text == "Hello there"  # last (final) partial carries empty text
    assert payloads[-1]["candidates"][0]["finishReason"] == "STOP"
    assert payloads[-1]["usageMetadata"]["totalTokenCount"] == 14


async def test_gemini_auth_via_query_param(client, mock_engine):
    r = await client.post(f"/v1beta/models/gemini-1.5-pro:generateContent?key={TEST_KEY}",
                          json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]})
    assert r.status_code == 200


async def test_gemini_auth_401(client, mock_engine):
    r = await client.post("/v1beta/models/gemini-1.5-pro:generateContent",
                          json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]})
    assert r.status_code == 401
    assert r.json()["error"]["status"] == "UNAUTHENTICATED"


async def test_gemini_inline_data_reaches_engine(client, mock_engine):
    await client.post("/v1beta/models/gemini-1.5-pro:generateContent", headers=AUTH_G, json={
        "contents": [{"role": "user", "parts": [
            {"text": "what is this?"},
            {"inline_data": {"mime_type": "image/png", "data": PNG_B64}}]}]})
    blocks = mock_engine["req"].messages[-1].blocks
    img = [b for b in blocks if b["type"] == "image"]
    assert img and img[0]["media_type"] == "image/png" and img[0]["data"] == PNG_B64


async def test_gemini_pdf_inline_data_becomes_document_block(client, mock_engine):
    await client.post("/v1beta/models/gemini-1.5-pro:generateContent", headers=AUTH_G, json={
        "contents": [{"role": "user", "parts": [
            {"text": "extract this"},
            {"inline_data": {"mime_type": "application/pdf", "data": PDF_B64}}]}]})
    blocks = mock_engine["req"].messages[-1].blocks
    doc = [b for b in blocks if b["type"] == "document"]
    assert doc and doc[0]["media_type"] == "application/pdf" and doc[0]["data"] == PDF_B64
    assert not [b for b in blocks if b["type"] == "image"]  # PDF must not become an image


async def test_gemini_normalizes_urlsafe_base64(client, mock_engine):
    # The official google-genai SDK encodes inline bytes with URL-safe base64.
    raw = b"\xfb\xff\xbf"  # standard base64 of this contains '+' and '/'
    urlsafe = base64.urlsafe_b64encode(raw).decode()
    assert "-" in urlsafe or "_" in urlsafe
    await client.post("/v1beta/models/gemini-1.5-pro:generateContent", headers=AUTH_G, json={
        "contents": [{"role": "user", "parts": [
            {"inline_data": {"mime_type": "image/png", "data": urlsafe}}]}]})
    block = mock_engine["req"].messages[-1].blocks[0]
    # Stored as canonical standard base64 (what the CLI/Anthropic API requires).
    assert block["data"] == base64.b64encode(raw).decode()
    assert "-" not in block["data"] and "_" not in block["data"]


async def test_gemini_nonstring_inline_data_is_rejected(client, mock_engine):
    # A non-string inline_data.data must surface as a 400 with the Gemini error
    # envelope, not crash with an AttributeError that escapes as a 500.
    r = await client.post("/v1beta/models/gemini-1.5-pro:generateContent", headers=AUTH_G, json={
        "contents": [{"role": "user", "parts": [
            {"inline_data": {"mime_type": "image/png", "data": 12345}}]}]})
    assert r.status_code == 400
    assert r.json() == {
        "error": {"code": 400, "message": "invalid base64 data", "status": "INVALID_ARGUMENT"}}


async def test_gemini_empty_inline_data_is_rejected(client, mock_engine):
    # Empty data is a clean 400 at the gateway boundary, not an empty media block
    # forwarded to the CLI.
    r = await client.post("/v1beta/models/gemini-1.5-pro:generateContent", headers=AUTH_G, json={
        "contents": [{"role": "user", "parts": [
            {"inline_data": {"mime_type": "image/png", "data": ""}}]}]})
    assert r.status_code == 400
    assert r.json() == {
        "error": {"code": 400, "message": "empty base64 data", "status": "INVALID_ARGUMENT"}}


async def test_gemini_unsupported_inline_data_mime_is_rejected(client, mock_engine):
    # Non-image, non-PDF inline media is rejected with a 400 instead of becoming a
    # malformed image block the CLI would reject.
    r = await client.post("/v1beta/models/gemini-1.5-pro:generateContent", headers=AUTH_G, json={
        "contents": [{"role": "user", "parts": [
            {"inline_data": {"mime_type": "audio/mpeg", "data": PNG_B64}}]}]})
    assert r.status_code == 400
    assert r.json()["error"]["status"] == "INVALID_ARGUMENT"


async def test_gemini_pdf_mime_with_params_routes_to_document(client, mock_engine):
    # Casing and a charset parameter must not break PDF detection.
    await client.post("/v1beta/models/gemini-1.5-pro:generateContent", headers=AUTH_G, json={
        "contents": [{"role": "user", "parts": [
            {"inline_data": {"mime_type": "Application/PDF; charset=binary", "data": PDF_B64}}]}]})
    blocks = mock_engine["req"].messages[-1].blocks
    doc = [b for b in blocks if b["type"] == "document"]
    assert doc and doc[0]["media_type"] == "application/pdf"


async def test_gemini_system_instruction(client, mock_engine):
    await client.post("/v1beta/models/gemini-1.5-pro:generateContent", headers=AUTH_G, json={
        "systemInstruction": {"parts": [{"text": "Be terse."}]},
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}]})
    assert mock_engine["req"].system == "Be terse."


async def test_gemini_models_list(client, mock_engine):
    r = await client.get("/v1beta/models", headers=AUTH_G)
    assert r.status_code == 200
    assert all(m["name"].startswith("models/") for m in r.json()["models"])


# ====================== error propagation ======================

async def test_anthropic_engine_error_envelope(client, mock_engine):
    mock_engine["events"] = [Error(502, "upstream boom")]
    r = await client.post("/v1/messages", headers=AUTH_A, json={
        "model": "claude-sonnet-4-6", "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 502
    assert r.json()["error"]["message"] == "upstream boom"


async def test_unknown_model_does_not_error(client, mock_engine):
    r = await client.post("/v1/chat/completions", headers=AUTH_O, json={
        "model": "some-future-model", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    # resolver fell back to default; engine still invoked
    assert mock_engine["req"].model == "sonnet"
