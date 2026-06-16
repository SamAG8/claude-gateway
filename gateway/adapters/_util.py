"""Small shared helpers for adapters."""
import json
import uuid

from fastapi import Request

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def bearer_token(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def sse(data: dict, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data)}\n\n"
