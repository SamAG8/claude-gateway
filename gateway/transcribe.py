"""Meeting recording -> transcript, for Nimbus's Meeting Minutes tool.

The gateway's only audio path, and deliberately narrow. A recording (usually an
MP4 of a ~30 minute site meeting, sometimes hundreds of MB) is reduced by ffmpeg
to mono 16 kHz speech at 32 kbps MP3 — about 7 MB for 30 minutes — and sent to an
audio-capable OpenRouter model as one ``input_audio`` part. The transcript comes
back in the language that was spoken; turning it into English minutes is the
client's next step, so the meeting is translated once, not twice.

Long recordings are cut into segments first, so no single answer can run into
the model's output cap and silently lose the end of the meeting.

No FastAPI here: the adapter owns the upload and the HTTP shape; this owns the
subprocesses and the upstream call, so tests can drive each without the other.
"""
import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from . import config, lanes
from .engines import openrouter

logger = logging.getLogger("claude-gateway.transcribe")

LANE = "http"

SYSTEM_PROMPT = (
    "You transcribe recordings of construction site meetings.\n"
    "Write a faithful, complete transcript of everything said, from start to end. "
    "Do not summarise, skip, or reorder anything.\n"
    "Keep each utterance in the language it was spoken (Persian stays Persian, "
    "English stays English); do not translate.\n"
    "Start a new line at each change of speaker, prefixed with a speaker label. "
    "Use a person's name when the conversation makes it clear (for example they "
    "are addressed by name or introduce themselves), otherwise 'Speaker 1', "
    "'Speaker 2', and so on, kept consistent throughout.\n"
    "Add a timestamp like [05:00] roughly every five minutes.\n"
    "Output only the transcript. If there is no intelligible speech, output "
    "exactly: [no speech]"
)


class TranscribeError(Exception):
    """A failure with the HTTP status the client should see."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class TranscriptResult:
    text: str
    duration_seconds: float
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    segments: int = 1
    notes: list[str] = field(default_factory=list)


async def _run(*args: str, timeout: float) -> tuple[int, bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise TranscribeError(504, "audio extraction took too long")
    return proc.returncode or 0, out or b"", err or b""


async def probe_duration(src: Path) -> float:
    """Seconds of audio in ``src``. 422 when there is no audio stream at all."""
    code, out, _ = await _run(
        config.FFPROBE_BIN, "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=duration:format=duration",
        "-of", "json", str(src), timeout=60)
    if code != 0:
        raise TranscribeError(422, "that file could not be read as a recording")
    try:
        data = json.loads(out.decode("utf-8") or "{}")
    except ValueError:
        raise TranscribeError(422, "that file could not be read as a recording")
    if not data.get("streams"):
        raise TranscribeError(422, "that recording has no sound")
    for raw in (data["streams"][0].get("duration"), (data.get("format") or {}).get("duration")):
        try:
            seconds = float(raw)
        except (TypeError, ValueError):
            continue
        if seconds > 0:
            return seconds
    raise TranscribeError(422, "that recording has no sound")


async def extract_audio(src: Path, workdir: Path, duration: float) -> list[Path]:
    """Mono 16 kHz 32 kbps MP3, cut into segments when the recording is long."""
    common = ["-nostdin", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
              "-c:a", "libmp3lame", "-b:a", "32k"]
    seg = config.TRANSCRIBE_SEGMENT_SECONDS
    if duration > seg * 1.25:
        pattern = workdir / "part-%03d.mp3"
        args = [config.FFMPEG_BIN, *common, "-f", "segment", "-segment_time", str(seg),
                "-reset_timestamps", "1", str(pattern)]
    else:
        args = [config.FFMPEG_BIN, *common, str(workdir / "part-000.mp3")]
    code, _, err = await _run(*args, timeout=max(120.0, duration / 4))
    if code != 0:
        logger.info("ffmpeg failed: %s", err[-400:].decode("utf-8", "replace"))
        raise TranscribeError(422, "the sound could not be taken out of that recording")
    parts = sorted(workdir.glob("part-*.mp3"))
    if not parts:
        raise TranscribeError(422, "the sound could not be taken out of that recording")
    return parts


def build_body(audio_b64: str, segment: int = 1, total: int = 1) -> dict:
    text = "Transcribe this recording."
    if total > 1:
        text = (f"This is part {segment} of {total} of one meeting recording. "
                "Transcribe this part; timestamps start at [00:00] for this part.")
    return {
        "model": config.TRANSCRIBE_MODEL,
        "stream": False,
        "max_tokens": config.TRANSCRIBE_MAX_TOKENS,
        "temperature": 0,
        # Transcription needs no deliberation, and thinking tokens are billed.
        "reasoning": {"effort": "low", "exclude": True},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}},
            ]},
        ],
    }


async def _transcribe_part(part: Path, segment: int, total: int) -> dict:
    audio_b64 = base64.b64encode(part.read_bytes()).decode("ascii")
    client = await openrouter.get_client()
    try:
        resp = await client.post("/chat/completions", headers=openrouter._headers(),
                                 json=build_body(audio_b64, segment, total),
                                 timeout=config.TRANSCRIBE_TIMEOUT)
    except Exception as e:  # noqa: BLE001 - timeouts and connection failures alike
        name = type(e).__name__
        status = 504 if "Timeout" in name else 502
        raise TranscribeError(status, f"the transcription service did not answer ({name})")
    if resp.status_code != 200:
        logger.info("transcribe upstream %s: %s", resp.status_code, resp.text[:400])
        status = 429 if resp.status_code == 429 else 502
        raise TranscribeError(status, f"the transcription service refused ({resp.status_code})")
    try:
        data = resp.json()
        choice = data["choices"][0]
        text = (choice.get("message") or {}).get("content") or ""
    except (ValueError, KeyError, IndexError, TypeError):
        raise TranscribeError(502, "the transcription service answered in an unexpected shape")
    if isinstance(text, list):  # some providers answer with content parts
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    return {"text": str(text).strip(), "finish": choice.get("finish_reason"),
            "usage": data.get("usage") or {}}


async def transcribe(src: Path, workdir: Path) -> TranscriptResult:
    """The whole path: probe, extract, transcribe each segment, join."""
    duration = await probe_duration(src)
    if duration > config.TRANSCRIBE_MAX_SECONDS:
        raise TranscribeError(
            422, f"that recording is longer than {config.TRANSCRIBE_MAX_SECONDS // 60} minutes")
    parts = await extract_audio(src, workdir, duration)

    try:
        await lanes.acquire(LANE)
    except lanes.Saturated:
        raise TranscribeError(503, "the transcription service is busy; try again shortly")
    try:
        results = []
        for i, part in enumerate(parts, start=1):
            results.append(await _transcribe_part(part, i, len(parts)))
    finally:
        lanes.release(LANE)

    result = TranscriptResult(text="", duration_seconds=round(duration, 1),
                              model=config.TRANSCRIBE_MODEL, segments=len(parts))
    pieces = []
    for i, r in enumerate(results, start=1):
        usage = r["usage"]
        result.input_tokens += int(usage.get("prompt_tokens") or 0)
        result.output_tokens += int(usage.get("completion_tokens") or 0)
        if usage.get("cost") is not None:
            result.cost_usd = (result.cost_usd or 0.0) + float(usage["cost"])
        if r["finish"] == "length":
            result.notes.append(f"part {i} of {len(parts)} was cut off before its end")
        if len(parts) > 1:
            start = (i - 1) * config.TRANSCRIBE_SEGMENT_SECONDS
            pieces.append(f"[part {i} of {len(parts)}, from {start // 60:02d}:00]\n{r['text']}")
        else:
            pieces.append(r["text"])
    result.text = "\n\n".join(pieces).strip()
    if not result.text:
        raise TranscribeError(502, "the transcription came back empty")
    return result
