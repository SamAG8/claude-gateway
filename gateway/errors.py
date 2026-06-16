"""Per-protocol error envelopes + constant-time API-key auth.

Each adapter returns its upstream's native error shape so SDKs parse failures
exactly as they would from the real service.
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


# --- Anthropic -----------------------------------------------------------

_ANTHROPIC_TYPES = {
    400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
    404: "not_found_error", 413: "request_too_large", 429: "rate_limit_error",
    500: "api_error", 502: "api_error", 504: "api_error",
}


def anthropic_error(status: int, message: str, err_type: str | None = None) -> JSONResponse:
    t = err_type or _ANTHROPIC_TYPES.get(status, "api_error")
    return JSONResponse(status_code=status,
                        content={"type": "error", "error": {"type": t, "message": message}})


# --- OpenAI --------------------------------------------------------------

_OPENAI_TYPES = {
    400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
    404: "not_found_error", 413: "invalid_request_error", 429: "rate_limit_error",
    500: "api_error", 502: "api_error", 504: "api_error",
}


def openai_error(status: int, message: str, err_type: str | None = None, code=None) -> JSONResponse:
    t = err_type or _OPENAI_TYPES.get(status, "api_error")
    return JSONResponse(status_code=status,
                        content={"error": {"message": message, "type": t, "param": None, "code": code}})


def openai_error_dict(status: int, message: str) -> dict:
    """Same envelope as a dict, for embedding in an SSE error chunk."""
    return {"error": {"message": message, "type": _OPENAI_TYPES.get(status, "api_error"),
                      "param": None, "code": None}}


# --- Gemini --------------------------------------------------------------

_GEMINI_STATUS = {
    400: "INVALID_ARGUMENT", 401: "UNAUTHENTICATED", 403: "PERMISSION_DENIED",
    404: "NOT_FOUND", 413: "INVALID_ARGUMENT", 429: "RESOURCE_EXHAUSTED",
    500: "INTERNAL", 502: "UNAVAILABLE", 504: "UNAVAILABLE",
}


def gemini_error(status: int, message: str, status_str: str | None = None) -> JSONResponse:
    s = status_str or _GEMINI_STATUS.get(status, "INTERNAL")
    return JSONResponse(status_code=status,
                        content={"error": {"code": status, "message": message, "status": s}})
