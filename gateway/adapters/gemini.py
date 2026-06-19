"""Google Gemini adapter — /v1beta/models/{model}:generate*/stream*, GET /v1beta/models (§9c)."""
from typing import Iterable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .. import models, protocol
from ..canonical import CanonicalMessage, CanonicalRequest, Delta, Error, Result, Start, Stop, map_reason
from ..content import document_block, image_block
from ..errors import GatewayError, gemini_error, key_is_valid
from ..models import resolve_model
from ..translate import join_texts, to_role
from ._util import sse

router = APIRouter()

_FINISH = {"end_turn": "STOP", "max_tokens": "MAX_TOKENS"}


def _finish(reason: str) -> str:
    return map_reason(_FINISH, reason, "OTHER")


def _api_key(request: Request) -> str | None:
    return request.headers.get("x-goog-api-key") or request.query_params.get("key")


def _system(si) -> str | None:
    if not si:
        return None
    return join_texts(si.get("parts", []))


def _to_messages(contents) -> list[CanonicalMessage]:
    out = []
    for c in contents:
        blocks: list[dict] = []
        for p in c.get("parts", []):
            if "text" in p:
                blocks.append({"type": "text", "text": p["text"]})
            else:
                inline = p.get("inline_data") or p.get("inlineData")
                if inline:
                    media_type = inline.get("mime_type") or inline.get("mimeType")
                    data = inline.get("data", "")
                    mt = (media_type or "").split(";")[0].strip().lower()
                    if mt == "application/pdf":
                        blocks.append(document_block(mt, data))
                    elif mt.startswith("image/"):
                        blocks.append(image_block(mt, data))
                    else:
                        raise GatewayError(400, f"unsupported inline_data mime type: {media_type!r}")
        out.append(CanonicalMessage(role=to_role(c.get("role"), ("model", "assistant")), blocks=blocks))
    return out


def _build(model_name: str, body: dict, stream: bool) -> CanonicalRequest:
    contents = body.get("contents")
    if not isinstance(contents, list) or not contents:
        raise GatewayError(400, "contents is required")
    gen = body.get("generationConfig") or {}
    return CanonicalRequest(
        model=resolve_model(model_name),
        requested_model=model_name,
        system=_system(body.get("systemInstruction") or body.get("system_instruction")),
        messages=_to_messages(contents),
        max_tokens=gen.get("maxOutputTokens"),
        stream=stream,
        temperature=gen.get("temperature"),
        top_p=gen.get("topP"),
        top_k=gen.get("topK"),
    )


def _unauthorized(request: Request) -> JSONResponse | None:
    if not key_is_valid(_api_key(request)):
        return gemini_error(401, "missing or invalid API key")
    return None


@router.get("/v1beta/models")
async def list_models(request: Request):
    if (resp := _unauthorized(request)) is not None:
        return resp
    return JSONResponse(models.gemini_models_payload())


@router.post("/v1beta/models/{model_method:path}")
async def generate(model_method: str, request: Request):
    if (resp := _unauthorized(request)) is not None:
        return resp
    if ":" not in model_method:
        return gemini_error(400, "expected models/{model}:{method}")
    model_name, method = model_method.rsplit(":", 1)
    if method not in ("generateContent", "streamGenerateContent"):
        return gemini_error(404, f"unknown method: {method}")
    try:
        body = await request.json()
    except Exception:
        return gemini_error(400, "invalid JSON body")
    stream = method == "streamGenerateContent"
    try:
        req = _build(model_name, body, stream)
    except GatewayError as e:
        return gemini_error(e.status, e.message)
    return await protocol.respond(req, _Formatter(req))


def _usage(prompt: int, completion: int) -> dict:
    return {"promptTokenCount": prompt, "candidatesTokenCount": completion,
            "totalTokenCount": prompt + completion}


class _Formatter:
    def __init__(self, req: CanonicalRequest):
        self.model = req.requested_model
        self.prompt = 0

    def on_start(self, ev: Start) -> Iterable[str]:
        self.model = ev.model or self.model
        self.prompt = ev.input_tokens
        return ()  # Gemini emits nothing until it has content

    def on_delta(self, ev: Delta) -> Iterable[str]:
        yield sse({"candidates": [{
            "content": {"role": "model", "parts": [{"text": ev.text}]}, "index": 0}]})

    def on_stop(self, ev: Stop) -> Iterable[str]:
        prompt = ev.input_tokens or self.prompt
        yield sse({
            "candidates": [{"content": {"role": "model", "parts": [{"text": ""}]},
                            "finishReason": _finish(ev.stop_reason), "index": 0}],
            "usageMetadata": _usage(prompt, ev.output_tokens),
            "modelVersion": self.model,
        })

    def on_error(self, ev: Error) -> Iterable[str]:
        yield sse({"error": {"code": ev.status, "message": ev.message, "status": "INTERNAL"}})

    def complete(self, result: Result) -> dict:
        return {
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": result.text}]},
                "finishReason": _finish(result.stop_reason),
                "index": 0,
            }],
            "usageMetadata": _usage(result.input_tokens, result.output_tokens),
            "modelVersion": result.model,
        }

    def error_response(self, status: int, message: str) -> JSONResponse:
        return gemini_error(status, message)
