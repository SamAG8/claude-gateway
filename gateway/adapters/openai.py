"""OpenAI Chat Completions adapter — POST /v1/chat/completions, GET /v1/models (§9b)."""
import time
from typing import Iterable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .. import models, protocol
from ..canonical import CanonicalMessage, CanonicalRequest, Delta, Error, Result, Start, Stop, map_reason
from ..content import image_block, parse_data_uri
from ..errors import GatewayError, key_is_valid, openai_error, openai_error_dict
from ..models import resolve_model
from ..translate import join_texts, to_role
from ._util import bearer_token, gen_id, sse

router = APIRouter()

_FINISH = {"end_turn": "stop", "max_tokens": "length"}


def _finish(reason: str) -> str:
    return map_reason(_FINISH, reason, "stop")


def _content_blocks(content) -> list[dict]:
    blocks: list[dict] = []
    if isinstance(content, str):
        blocks.append({"type": "text", "text": content})
    elif isinstance(content, list):
        for part in content:
            pt = part.get("type")
            if pt == "text":
                blocks.append({"type": "text", "text": part.get("text", "")})
            elif pt == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                media_type, data = parse_data_uri(url)
                blocks.append(image_block(media_type, data))
    return blocks


def _build(body: dict) -> CanonicalRequest:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise GatewayError(400, "messages is required")
    system_parts: list[dict] = []
    canon: list[CanonicalMessage] = []
    for m in messages:
        if m.get("role") == "system":
            content = m.get("content")
            text = content if isinstance(content, str) else join_texts(content or [], sep=" ") or ""
            system_parts.append({"text": text})
            continue
        canon.append(CanonicalMessage(role=to_role(m.get("role")),
                                      blocks=_content_blocks(m.get("content"))))
    if not canon:
        raise GatewayError(400, "at least one non-system message is required")
    requested = body.get("model", "") or "gpt-4o"
    return CanonicalRequest(
        model=resolve_model(requested),
        requested_model=requested,
        system=join_texts(system_parts),
        messages=canon,
        max_tokens=body.get("max_tokens"),
        stream=bool(body.get("stream", False)),
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        stop=body.get("stop"),
        tools=body.get("tools"),
    )


def _unauthorized(request: Request) -> JSONResponse | None:
    if not key_is_valid(bearer_token(request)):
        return openai_error(401, "invalid Authorization bearer token", "authentication_error")
    return None


@router.get("/v1/models")
async def list_models(request: Request):
    if (resp := _unauthorized(request)) is not None:
        return resp
    return JSONResponse(models.openai_models_payload())


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if (resp := _unauthorized(request)) is not None:
        return resp
    try:
        body = await request.json()
    except Exception:
        return openai_error(400, "invalid JSON body")
    try:
        req = _build(body)
    except GatewayError as e:
        return openai_error(e.status, e.message, e.err_type)
    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
    return await protocol.respond(req, _Formatter(req, include_usage))


class _Formatter:
    def __init__(self, req: CanonicalRequest, include_usage: bool = False):
        self.cid = gen_id("chatcmpl")
        self.created = int(time.time())
        self.model = req.requested_model
        self.include_usage = include_usage
        self.prompt = 0
        self._role_sent = False

    def _chunk(self, delta: dict, finish=None) -> str:
        return sse({"id": self.cid, "object": "chat.completion.chunk", "created": self.created,
                    "model": self.model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

    def on_start(self, ev: Start) -> Iterable[str]:
        self.model = ev.model or self.model
        self.prompt = ev.input_tokens
        self._role_sent = True
        yield self._chunk({"role": "assistant"})

    def on_delta(self, ev: Delta) -> Iterable[str]:
        if not self._role_sent:  # defensive: role chunk must precede content
            self._role_sent = True
            yield self._chunk({"role": "assistant"})
        yield self._chunk({"content": ev.text})

    def on_stop(self, ev: Stop) -> Iterable[str]:
        prompt = ev.input_tokens or self.prompt
        completion = ev.output_tokens
        yield self._chunk({}, finish=_finish(ev.stop_reason))
        if self.include_usage:
            yield sse({"id": self.cid, "object": "chat.completion.chunk", "created": self.created,
                       "model": self.model, "choices": [],
                       "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                                 "total_tokens": prompt + completion}})
        yield "data: [DONE]\n\n"

    def on_error(self, ev: Error) -> Iterable[str]:
        yield sse(openai_error_dict(ev.status, ev.message))
        yield "data: [DONE]\n\n"

    def complete(self, result: Result) -> dict:
        prompt, completion = result.input_tokens, result.output_tokens
        return {
            "id": self.cid,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,  # echo the requested model id (e.g. gpt-4o)
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": _finish(result.stop_reason),
            }],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                      "total_tokens": prompt + completion},
        }

    def error_response(self, status: int, message: str) -> JSONResponse:
        return openai_error(status, message)
