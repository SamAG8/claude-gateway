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


# CanonicalEvent: the typed contract the engine yields to every adapter. The four
# kinds form a tagged union; adapters dispatch on type, not on a string key.
@dataclass
class Start:
    model: Optional[str]
    input_tokens: int


@dataclass
class Delta:
    text: str


@dataclass
class Stop:
    stop_reason: str          # "end_turn" | "max_tokens" | "error"
    output_tokens: int
    input_tokens: int


@dataclass
class Error:
    status: int
    message: str


CanonicalEvent = Start | Delta | Stop | Error


@dataclass
class Result:
    """The drained, non-streaming outcome of one invocation (engine.collect)."""
    text: str
    model: str
    stop_reason: str
    input_tokens: int
    output_tokens: int
    error: Optional[Error] = None


def map_stop_reason(cli_reason: Optional[str], is_error: bool = False) -> str:
    """Map a CLI stop_reason to the canonical set: end_turn / max_tokens / error."""
    if is_error or cli_reason not in ("end_turn", "max_tokens"):
        return "error"
    return cli_reason


def map_reason(table: dict[str, str], reason: str, default: str) -> str:
    """Shared shape for every adapter's canonical→native stop/finish mapping."""
    return table.get(reason, default)
