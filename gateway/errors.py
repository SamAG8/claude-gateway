"""Per-protocol error envelopes + constant-time API-key auth.

The HTTP-status taxonomy lives in one place (`_STATUS_KIND`); each protocol is a
thin adapter that maps a canonical kind to its own string and shapes the envelope.
"""
import hmac

from fastapi.responses import JSONResponse

from . import config


class GatewayError(Exception):
    """Raised inside adapters; carries an HTTP status + message to render natively."""

    def __init__(self, status: int, message: str, err_type: str | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.err_type = err_type


def key_is_valid(provided: str | None) -> bool:
    """Constant-time compare of a presented key against the configured set."""
    if not provided or not config.API_KEYS:
        return False
    ok = False
    for k in config.API_KEYS:
        if hmac.compare_digest(provided, k):
            ok = True
    return ok


# Canonical meaning of each HTTP status the gateway emits — the single taxonomy.
# "server" (500) and "unavailable" (502/504) are distinct: some protocols name them differently.
_STATUS_KIND = {
    400: "invalid_request", 401: "auth", 403: "permission", 404: "not_found",
    413: "too_large", 429: "rate_limit", 500: "server", 502: "unavailable", 504: "unavailable",
}


def _kind(status: int) -> str:
    return _STATUS_KIND.get(status, "server")


# Each protocol maps the canonical kind to its own type string.
_ANTHROPIC = {
    "invalid_request": "invalid_request_error", "auth": "authentication_error",
    "permission": "permission_error", "not_found": "not_found_error",
    "too_large": "request_too_large", "rate_limit": "rate_limit_error",
    "server": "api_error", "unavailable": "api_error",
}
_OPENAI = {
    "invalid_request": "invalid_request_error", "auth": "authentication_error",
    "permission": "permission_error", "not_found": "not_found_error",
    "too_large": "invalid_request_error", "rate_limit": "rate_limit_error",
    "server": "api_error", "unavailable": "api_error",
}
_GEMINI = {
    "invalid_request": "INVALID_ARGUMENT", "auth": "UNAUTHENTICATED",
    "permission": "PERMISSION_DENIED", "not_found": "NOT_FOUND",
    "too_large": "INVALID_ARGUMENT", "rate_limit": "RESOURCE_EXHAUSTED",
    "server": "INTERNAL", "unavailable": "UNAVAILABLE",
}


def anthropic_error(status: int, message: str, err_type: str | None = None) -> JSONResponse:
    t = err_type or _ANTHROPIC[_kind(status)]
    return JSONResponse(status_code=status,
                        content={"type": "error", "error": {"type": t, "message": message}})


def openai_error(status: int, message: str, err_type: str | None = None, code=None) -> JSONResponse:
    return JSONResponse(status_code=status, content=openai_error_dict(status, message, err_type, code))


def openai_error_dict(status: int, message: str, err_type: str | None = None, code=None) -> dict:
    """The OpenAI envelope as a dict, for embedding in an SSE error chunk."""
    t = err_type or _OPENAI[_kind(status)]
    return {"error": {"message": message, "type": t, "param": None, "code": code}}


def gemini_error(status: int, message: str, status_str: str | None = None) -> JSONResponse:
    s = status_str or _GEMINI[_kind(status)]
    return JSONResponse(status_code=status,
                        content={"error": {"code": status, "message": message, "status": s}})
