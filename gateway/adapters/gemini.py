"""Google Gemini adapter — /v1beta/models/{model}:generate*/stream*, GET /v1beta/models (§9c)."""
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .. import engine, models
from ..canonical import CanonicalMessage, CanonicalRequest
from ..content import image_block
from ..errors import GatewayError, gemini_error, key_is_valid
from ..models import resolve_model
from ._util import SSE_HEADERS, sse

router = APIRouter()

_FINISH = {"end_turn": "STOP", "max_tokens": "MAX_TOKENS"}


def _finish(reason: str) -> str:
    return _FINISH.get(reason, "OTHER")


def _api_key(request: Request) -> str | None:
    return request.headers.get("x-goog-api-key") or request.query_params.get("key")


def _system(si) -> str | None:
    if not si:
        return None
    parts = si.get("parts", [])
    return "\n".join(p.get("text", "") for p in parts if "text" in p) or None


def _to_messages(contents) -> list[CanonicalMessage]:
    out = []
    for c in contents:
        role = "assistant" if c.get("role") == "model" else "user"
        blocks: list[dict] = []
        for p in c.get("parts", []):
            if "text" in p:
                blocks.append({"type": "text", "text": p["text"]})
            else:
                inline = p.get("inline_data") or p.get("inlineData")
                if inline:
                    media_type = inline.get("mime_type") or inline.get("mimeType")
                    blocks.append(image_block(media_type, inline.get("data", "")))
        out.append(CanonicalMessage(role=role, blocks=blocks))
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


def _usage(prompt: int, completion: int) -> dict:
    return {"promptTokenCount": prompt, "candidatesTokenCount": completion,
            "totalTokenCount": prompt + completion}


@router.get("/v1beta/models")
async def list_models(request: Request):
    if not key_is_valid(_api_key(request)):
        return gemini_error(401, "missing or invalid API key")
    return JSONResponse(models.gemini_models_payload())


@router.post("/v1beta/models/{model_method:path}")
async def generate(model_method: str, request: Request):
    if not key_is_valid(_api_key(request)):
        return gemini_error(401, "missing or invalid API key")
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

    if stream:
        return StreamingResponse(_stream(req), media_type="text/event-stream", headers=SSE_HEADERS)
    return await _complete(req)


async def _complete(req: CanonicalRequest):
    out = await engine.collect(req)
    if out["error"]:
        return gemini_error(out["error"]["status"], out["error"]["message"])
    return JSONResponse({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": out["text"]}]},
            "finishReason": _finish(out["stop_reason"]),
            "index": 0,
        }],
        "usageMetadata": _usage(out["input_tokens"], out["output_tokens"]),
        "modelVersion": out["model"],
    })


async def _stream(req: CanonicalRequest):
    model = req.requested_model
    prompt = completion = 0
    async for ev in engine.run_claude(req):
        t = ev["t"]
        if t == "start":
            model = ev.get("model") or model
            prompt = ev.get("input_tokens", 0)
        elif t == "delta":
            yield sse({"candidates": [{
                "content": {"role": "model", "parts": [{"text": ev["text"]}]}, "index": 0}]})
        elif t == "stop":
            completion = ev.get("output_tokens", 0)
            prompt = ev.get("input_tokens", prompt)
            yield sse({
                "candidates": [{"content": {"role": "model", "parts": [{"text": ""}]},
                                "finishReason": _finish(ev["stop_reason"]), "index": 0}],
                "usageMetadata": _usage(prompt, completion),
                "modelVersion": model,
            })
            return
        elif t == "error":
            yield sse({"error": {"code": ev["status"], "message": ev["message"], "status": "INTERNAL"}})
            return
