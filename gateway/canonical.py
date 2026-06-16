"""The internal contract every adapter speaks to the core engine.

Adapters translate their protocol's request into a CanonicalRequest, call
``engine.run_claude``, and format the yielded CanonicalEvents back into their
protocol's response. The engine never imports an adapter.
"""
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CanonicalMessage:
    role: str  # "user" | "assistant"
    # Ordered content blocks:
    #   {"type": "text", "text": str}
    #   {"type": "image", "media_type": str, "data": <base64 str>}
    blocks: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CanonicalRequest:
    model: str               # resolved CLI --model value (after model-map resolution)
    requested_model: str     # the model string the client sent (echoed back in responses)
    system: Optional[str]    # merged plain system text, or None
    messages: list[CanonicalMessage] = field(default_factory=list)
    max_tokens: Optional[int] = None
    stream: bool = False
    # Accepted but ignored (CLI cannot set them) — retained for logging only.
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop: Optional[list] = None
    tools: Optional[Any] = None


# CanonicalEvent is a plain tagged dict yielded by the engine:
#   {"t": "start", "model": str, "input_tokens": int}
#   {"t": "delta", "text": str}
#   {"t": "stop",  "stop_reason": "end_turn"|"max_tokens"|"error", "output_tokens": int, "input_tokens": int}
#   {"t": "error", "status": int, "message": str}


def map_stop_reason(cli_reason: Optional[str], is_error: bool = False) -> str:
    """Map a CLI stop_reason to the canonical set: end_turn / max_tokens / error."""
    if is_error or cli_reason not in ("end_turn", "max_tokens"):
        return "error"
    return cli_reason
