"""POST /v1/audio/transcriptions: auth, org gate, limits, and the upstream body.

ffmpeg/ffprobe are faked at the subprocess boundary and OpenRouter at the socket,
so these run anywhere. One test at the bottom drives the real ffmpeg when the
machine has it, because the argument list is only proven by running it.
"""
import json
import shutil
from pathlib import Path

import httpx
import pytest

from gateway import config, introspect, transcribe
from gateway.engines import openrouter

from conftest import TEST_KEY

URL = "/v1/audio/transcriptions"


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(config, "API_KEYS", {TEST_KEY})
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(config, "transcribe_enabled", lambda: True)
    monkeypatch.setattr(config, "TRANSCRIBE_ORG_IDS", set())


@pytest.fixture
def fake_media(monkeypatch):
    """Fake ffprobe (reports ``state['duration']``) and ffmpeg (writes parts)."""
    state = {"duration": 1800.0, "calls": [], "ffmpeg_code": 0, "streams": True}

    async def fake_run(*args, timeout):
        state["calls"].append(list(args))
        if args[0] == config.FFPROBE_BIN:
            streams = [{"duration": str(state["duration"])}] if state["streams"] else []
            return 0, json.dumps({"streams": streams, "format": {}}).encode(), b""
        if state["ffmpeg_code"]:
            return state["ffmpeg_code"], b"", b"boom"
        target = Path(args[-1])
        if "%03d" in target.name:
            n = int(state["duration"] // config.TRANSCRIBE_SEGMENT_SECONDS) + 1
            for i in range(n):
                (target.parent / f"part-{i:03d}.mp3").write_bytes(b"ID3fake%d" % i)
        else:
            target.write_bytes(b"ID3fake")
        return 0, b"", b""

    monkeypatch.setattr(transcribe, "_run", fake_run)
    return state


@pytest.fixture
def upstream(monkeypatch):
    holder = {"bodies": [], "status": 200, "texts": ["Nami: Good morning.\nMaddy: Hi."],
              "finish": "stop"}

    def handler(request: httpx.Request) -> httpx.Response:
        holder["url"] = str(request.url)
        holder["headers"] = dict(request.headers)
        holder["bodies"].append(json.loads(request.content))
        if holder["status"] != 200:
            return httpx.Response(holder["status"], text="no")
        text = holder["texts"][min(len(holder["bodies"]), len(holder["texts"])) - 1]
        return httpx.Response(200, json={
            "choices": [{"message": {"content": text}, "finish_reason": holder["finish"]}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 50, "cost": 0.01}})

    client = httpx.AsyncClient(base_url="https://openrouter.test/api/v1",
                              transport=httpx.MockTransport(handler))

    async def fake_get_client():
        return client

    monkeypatch.setattr(openrouter, "get_client", fake_get_client)
    return holder


async def test_happy_path_sends_mp3_input_audio(client, enabled, fake_media, upstream):
    r = await client.post(URL, headers={"x-api-key": TEST_KEY}, content=b"\x00mp4bytes")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["text"].startswith("Nami: Good morning.")
    assert data["duration_seconds"] == 1800.0
    assert data["segments"] == 1
    assert data["usage"] == {"input_tokens": 1000, "output_tokens": 50, "cost_usd": 0.01}

    body = upstream["bodies"][0]
    assert body["model"] == config.TRANSCRIBE_MODEL
    assert body["stream"] is False
    part = body["messages"][1]["content"][1]
    assert part["type"] == "input_audio"
    assert part["input_audio"]["format"] == "mp3"
    assert upstream["url"].endswith("/chat/completions")
    assert upstream["headers"]["authorization"] == "Bearer sk-test"

    ffmpeg = next(c for c in fake_media["calls"] if c[0] == config.FFMPEG_BIN)
    assert ["-ac", "1"] == ffmpeg[ffmpeg.index("-ac"):ffmpeg.index("-ac") + 2]
    assert "-vn" in ffmpeg


async def test_long_recording_is_segmented_and_joined(client, enabled, fake_media, upstream):
    fake_media["duration"] = 70 * 60.0
    upstream["texts"] = ["first", "second", "third"]
    r = await client.post(URL, headers={"x-api-key": TEST_KEY}, content=b"x")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["segments"] == 3
    assert data["text"].index("first") < data["text"].index("second") < data["text"].index("third")
    assert "[part 2 of 3, from 30:00]" in data["text"]
    assert len(upstream["bodies"]) == 3


async def test_cut_off_answer_is_reported(client, enabled, fake_media, upstream):
    upstream["finish"] = "length"
    r = await client.post(URL, headers={"x-api-key": TEST_KEY}, content=b"x")
    assert r.json()["notes"] == ["part 1 of 1 was cut off before its end"]


async def test_bad_key_is_401(client, enabled, fake_media, upstream):
    r = await client.post(URL, headers={"x-api-key": "nope"}, content=b"x")
    assert r.status_code == 401


async def test_org_gate(client, enabled, fake_media, upstream, monkeypatch):
    monkeypatch.setattr(config, "TOKEN_INTROSPECT_URL", "http://introspect.test")
    monkeypatch.setattr(config, "TRANSCRIBE_ORG_IDS", {"org-ace"})

    async def fake_record(tok):
        return {"active": True, "org_id": "org-ace" if tok == "cap_ace" else "org-other"}

    monkeypatch.setattr(introspect, "introspection", fake_record)
    r = await client.post(URL, headers={"x-api-key": "cap_other"}, content=b"x")
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "app_access_denied"
    r = await client.post(URL, headers={"x-api-key": "cap_ace"}, content=b"x")
    assert r.status_code == 200


async def test_too_large_is_413(client, enabled, fake_media, upstream, monkeypatch):
    monkeypatch.setattr(config, "TRANSCRIBE_MAX_UPLOAD", 4)
    r = await client.post(URL, headers={"x-api-key": TEST_KEY}, content=b"123456789")
    assert r.status_code == 413
    assert upstream["bodies"] == []


async def test_empty_body_is_400(client, enabled, fake_media, upstream):
    r = await client.post(URL, headers={"x-api-key": TEST_KEY}, content=b"")
    assert r.status_code == 400


async def test_no_sound_is_422(client, enabled, fake_media, upstream):
    fake_media["streams"] = False
    r = await client.post(URL, headers={"x-api-key": TEST_KEY}, content=b"x")
    assert r.status_code == 422
    assert "no sound" in r.json()["error"]["message"]


async def test_too_long_is_422(client, enabled, fake_media, upstream, monkeypatch):
    monkeypatch.setattr(config, "TRANSCRIBE_MAX_SECONDS", 600)
    r = await client.post(URL, headers={"x-api-key": TEST_KEY}, content=b"x")
    assert r.status_code == 422


async def test_upstream_failure_is_502_and_releases_lane(client, enabled, fake_media, upstream):
    from gateway import lanes

    upstream["status"] = 500
    before = lanes.get_semaphore("http")._value
    r = await client.post(URL, headers={"x-api-key": TEST_KEY}, content=b"x")
    assert r.status_code == 502
    assert lanes.get_semaphore("http")._value == before


async def test_temp_files_are_removed(client, enabled, fake_media, upstream, monkeypatch, tmp_path):
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    await client.post(URL, headers={"x-api-key": TEST_KEY}, content=b"x")
    assert list(tmp_path.iterdir()) == []


async def test_not_configured_is_503(client, enabled, monkeypatch):
    monkeypatch.setattr(config, "transcribe_enabled", lambda: False)
    r = await client.post(URL, headers={"x-api-key": TEST_KEY}, content=b"x")
    assert r.status_code == 503


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg")
async def test_real_ffmpeg_extracts_small_mp3(tmp_path):
    import subprocess

    src = tmp_path / "in.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                    "-f", "lavfi", "-i", "color=c=black:s=64x64:d=3", "-shortest",
                    "-c:v", "libx264", "-c:a", "aac", str(src)],
                   check=True, capture_output=True)
    duration = await transcribe.probe_duration(src)
    assert 2.5 < duration < 3.5
    parts = await transcribe.extract_audio(src, tmp_path, duration)
    assert len(parts) == 1 and parts[0].stat().st_size > 0
