"""The internal contract every adapter speaks to the core engine.

Adapters translate their protocol's request into a CanonicalRequest, call
``engine.run``, and format the yielded CanonicalEvents back into their protocol's
response. The engine never imports an adapter, and an adapter never learns which
engine answered.
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
    surface: str = ""        # originating protocol ("anthropic"|"openai"|"gemini"); for usage logging only
    effort_override: Optional[str] = None  # optional ``name:effort`` request suffix
    messages: list[CanonicalMessage] = field(default_factory=list)
    max_tokens: Optional[int] = None
    stream: bool = False
    # Accepted but ignored (CLI cannot set them) — retained for logging only.
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop: Optional[list] = None
    tools: Optional[Any] = None
    # Per-user MCP token (from the x-mcp-token header). When set and MCP is enabled,
    # the CLI runs with the configured MCP server attached, authenticated as this user.
    mcp_token: Optional[str] = None
    # Which engine answers this request, and why it is not the one the model asked
    # for. Written ONLY by engine.select_engine — never by an adapter, which cannot
    # know the resolved model's engine and must not second-guess the constraints.
    engine: str = "cli"
    route_reason: Optional[str] = None
    # Milliseconds spent authenticating before any engine was chosen. Logged, not
    # acted on: it is the one pre-engine cost that can be a round trip, and
    # without it a slow turn cannot be attributed.
    introspect_ms: Optional[int] = None


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


def media_stats(req: "CanonicalRequest") -> tuple[int, int, int]:
    """Count native image/document blocks and approx decoded bytes across all turns.

    Only blocks still present as native media reach an engine — a PDF the
    Anthropic surface flattened to text (x-pdf-mode: text, or past MAX_PDF_PAGES)
    is not counted, so ``docs`` here reflects native-vision PDFs (the expensive
    path), not text-extracted ones.

    Called at LOG time, never before an invocation: it walks every turn, and the
    answer is wanted for accounting, not for any decision. Doing it on the way in
    put an O(history) loop in front of the first token for nobody's benefit.
    """
    imgs = docs = nbytes = 0
    for m in (req.messages or []):
        for b in m.blocks:
            t = b.get("type")
            if t == "image":
                imgs += 1
            elif t == "document":
                docs += 1
            else:
                continue
            nbytes += (len(b.get("data") or "") * 3) // 4  # base64 → approx raw bytes
    return imgs, docs, nbytes


def map_stop_reason(cli_reason: Optional[str], is_error: bool = False) -> str:
    """Map a CLI stop_reason to the canonical set: end_turn / max_tokens / error."""
    if is_error or cli_reason not in ("end_turn", "max_tokens"):
        return "error"
    return cli_reason


def map_reason(table: dict[str, str], reason: str, default: str) -> str:
    """Shared shape for every adapter's canonical→native stop/finish mapping."""
    return table.get(reason, default)
