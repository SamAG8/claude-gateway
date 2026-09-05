"""Google Gemini adapter — /v1beta/models/{model}:generate*/stream*, GET /v1beta/models (§9c)."""
import re
from typing import Iterable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .. import models, protocol
from ..canonical import CanonicalMessage, CanonicalRequest, Delta, Error, Result, Start, Stop, map_reason
from ..content import document_block, image_block
from ..errors import GatewayError, gemini_error, key_is_valid
from ..models import parse_model_spec
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


_JSON_MODE_INSTRUCTION = (
    "IMPORTANT: Respond with ONLY the raw JSON value requested — no markdown "
    "code fences, no explanation, no commentary before or after it. The "
    "entire response body must be valid JSON and nothing else."
)


def _wants_json(gen: dict) -> bool:
    return gen.get("responseMimeType") == "application/json" or "responseSchema" in gen


def _extract_json_text(text: str) -> str:
    r"""Best-effort: pull the JSON value out of a response that may still carry
    a ```json fence and/or trailing prose, despite the JSON-mode instruction.

    NON-STREAMING ONLY, and that is a limit rather than an oversight. Extracting
    a JSON value from a stream would mean buffering the whole response before
    emitting anything, which is the one thing a streaming caller asked not to
    happen. A streaming JSON-mode request still gets the system instruction
    above — the model is told to return raw JSON — it just does not get this
    second line of defence.

    The fence branch is first because it is the common case. The brace-matching
    fallback exists for a response with prose and no fence, where anchoring to
    the end of the string (`\s*```$`, which every downstream consumer used)
    matched nothing and json.loads then failed on the prose.
    """
    body = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", body, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    start = None
    for i, ch in enumerate(body):
        if ch in "{[":
            start = i
            break
    if start is None:
        return body
    open_ch, close_ch = body[start], "}" if body[start] == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(body)):
        ch = body[i]
        # STRING-AWARE, because a brace inside a string value is not structure.
        # Counting it truncated `{"note": "a } inside a string"}` at the quote
        # and returned INVALID JSON — the exact failure this function exists to
        # prevent. Construction text is full of stray braces and brackets.
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return body[start : i + 1]
    return body


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
    resolved, effort = parse_model_spec(model_name)
    system = _system(body.get("systemInstruction") or body.get("system_instruction"))
    if _wants_json(gen):
        system = f"{system}\n\n{_JSON_MODE_INSTRUCTION}" if system else _JSON_MODE_INSTRUCTION
    return CanonicalRequest(
        model=resolved,
        requested_model=model_name,
        effort_override=effort,
        surface="gemini",
        system=system,
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
    json_mode = _wants_json(body.get("generationConfig") or {})
    return await protocol.respond(req, _Formatter(req, json_mode=json_mode), request)


def _usage(prompt: int, completion: int) -> dict:
    return {"promptTokenCount": prompt, "candidatesTokenCount": completion,
            "totalTokenCount": prompt + completion}


class _Formatter:
    def __init__(self, req: CanonicalRequest, json_mode: bool = False):
        self.model = req.requested_model
        self.prompt = 0
        self.json_mode = json_mode

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
        text = _extract_json_text(result.text) if self.json_mode else result.text
        return {
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": text}]},
                "finishReason": _finish(result.stop_reason),
                "index": 0,
            }],
            "usageMetadata": _usage(result.input_tokens, result.output_tokens),
            "modelVersion": result.model,
        }

    def error_response(self, status: int, message: str) -> JSONResponse:
        return gemini_error(status, message)
