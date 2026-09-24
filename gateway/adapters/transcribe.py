"""POST /v1/audio/transcriptions — a meeting recording in, its transcript out.

The request BODY is the recording itself (``Content-Type: video/mp4`` or any
audio/video type ffmpeg reads), not a multipart form: a 500 MB form would be
parsed and spooled by the framework before this code saw a byte, while a raw
body streams straight to one temp file under a hard cap, and a browser can send
a File that way with upload progress.

Auth is the same two doors as /v1/messages — the shared key, or the user's own
ConstraAP PAT — plus one gate the chat path does not have: when
``TRANSCRIBE_ORG_IDS`` is set, the PAT must belong to one of those orgs. This is
billed per minute of audio, and it is an ACE tool.

Response: ``{"text", "duration_seconds", "model", "segments", "notes", "usage"}``.
Errors use the OpenAI envelope.
"""
import logging
import shutil
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, Request

from .. import config, introspect, transcribe
from ..errors import key_is_valid, openai_error
from ._util import bearer_token

logger = logging.getLogger("claude-gateway.transcribe")

router = APIRouter()


async def _authorize(request: Request):
    """None when allowed, else the error response."""
    key = request.headers.get("x-api-key") or bearer_token(request)
    if key_is_valid(key):
        return None
    record = await introspect.introspection(key) if config.pat_auth_enabled() else None
    if record is None:
        return openai_error(401, "invalid credentials", "authentication_error")
    if config.TRANSCRIBE_ORG_IDS and str(record.get("org_id") or "") not in config.TRANSCRIBE_ORG_IDS:
        return openai_error(403, "Meeting Minutes is not enabled for your organization",
                            "permission_error", code="app_access_denied")
    return None


@router.post("/v1/audio/transcriptions")
async def transcriptions(request: Request):
    err = await _authorize(request)
    if err is not None:
        return err
    if not config.transcribe_enabled():
        return openai_error(503, "transcription is not configured on this gateway")

    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > config.TRANSCRIBE_MAX_UPLOAD:
        return openai_error(413, "that recording is too large")

    started = time.monotonic()
    workdir = Path(tempfile.mkdtemp(prefix="gw-transcribe-"))
    try:
        src = workdir / "recording"
        size = 0
        with src.open("wb") as fh:
            async for chunk in request.stream():
                size += len(chunk)
                if size > config.TRANSCRIBE_MAX_UPLOAD:
                    return openai_error(413, "that recording is too large")
                fh.write(chunk)
        if size == 0:
            return openai_error(400, "the request carried no recording")
        try:
            result = await transcribe.transcribe(src, workdir)
        except transcribe.TranscribeError as e:
            logger.info("transcribe failed status=%s bytes=%s: %s", e.status, size, e.message)
            return openai_error(e.status, e.message)
        logger.info("transcribed bytes=%s seconds=%s segments=%s in=%s out=%s cost=%s elapsed=%.1fs",
                    size, result.duration_seconds, result.segments, result.input_tokens,
                    result.output_tokens, result.cost_usd, time.monotonic() - started)
        return {
            "text": result.text,
            "duration_seconds": result.duration_seconds,
            "model": result.model,
            "segments": result.segments,
            "notes": result.notes,
            "usage": {"input_tokens": result.input_tokens,
                      "output_tokens": result.output_tokens,
                      "cost_usd": result.cost_usd},
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
