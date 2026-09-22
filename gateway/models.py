"""Model-name resolution and model-list payload builders.

The map lives in an editable JSON file (config.MODELS_FILE) that is hot-reloaded
by mtime so ops can retune aliases without a restart. Unknown models never error
— they fall back to the default — and real Claude ids/aliases pass straight
through, so the gateway stays current as Claude's aliases track the latest models.
"""
import json
from pathlib import Path

from . import config

_DEFAULT_MAP = {
    "default": "sonnet",
    "aliases": {},
    "passthrough_prefixes": ["claude-"],
}

_cache: dict = {"mtime": None, "path": None, "data": None}


def _load() -> dict:
    path = Path(config.MODELS_FILE)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        if _cache["data"] is None:
            _cache["data"] = dict(_DEFAULT_MAP)
        return _cache["data"]

    if _cache["data"] is None or _cache["mtime"] != mtime or _cache["path"] != str(path):
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            data = dict(_DEFAULT_MAP)
        _cache.update(mtime=mtime, path=str(path), data=data)
    return _cache["data"]


def _default_model(data: dict) -> str:
    return config.DEFAULT_MODEL or data.get("default") or "sonnet"


def resolve_model(requested: str) -> str:
    """Resolve a client model string to a CLI --model value (never raises)."""
    data = _load()
    if not requested:
        return _default_model(data)
    # Canonical tier names are part of the public API even when an operator's
    # custom models file omits redundant self-aliases.
    if requested in {"sonnet", "opus", "haiku"}:
        return requested
    for prefix in data.get("passthrough_prefixes", []):
        if requested.startswith(prefix):
            return requested
    aliases = data.get("aliases", {})
    if requested in aliases:
        return aliases[requested]
    return _default_model(data)


def model_tier(resolved_model: str) -> str:
    """Return the stable haiku/sonnet/opus family for a resolved CLI model id.

    Dated ids pass through to the CLI, but operational policy must still inherit
    the family-level fast-lane, effort, and thinking-budget settings.
    """
    value = (resolved_model or "").lower()
    for tier in ("haiku", "sonnet", "opus"):
        if tier in value:
            return tier
    return resolved_model


def _per_model_value(section: str, resolved_model: str):
    values = _load().get(section, {})
    if resolved_model in values:
        return True, values[resolved_model]
    tier = model_tier(resolved_model)
    if tier in values:
        return True, values[tier]
    return False, None


def resolve_effort(resolved_model: str) -> str | None:
    """Effort for a resolved --model, or None to use the CLI default.

    Precedence: a per-model entry in the models file's ``effort`` map wins; else
    the global ``EFFORT`` env applies to every model. This lets the fast tier
    (haiku) run at low effort for latency-sensitive extraction while heavier
    models (opus, used by ConstraBid) keep the global setting — no cross-impact.
    """
    found, value = _per_model_value("effort", resolved_model)
    if found:
        return value or None
    return config.EFFORT or None


def resolve_max_thinking_tokens(resolved_model: str) -> int | None:
    """MAX_THINKING_TOKENS to inject for a resolved --model, or None to leave the
    CLI's default thinking budget untouched.

    Precedence: a per-model entry in the models file's ``max_thinking_tokens`` map
    wins; else the global ``MAX_THINKING_TOKENS`` config applies. Only the fast tier
    (haiku, keyed as ``{"haiku": 0}``) is present by default, so heavier models
    (opus/sonnet used for extraction) get None and keep their thinking budget — no
    cross-impact. A value of 0 fully disables extended thinking in the CLI.
    """
    found, value = _per_model_value("max_thinking_tokens", resolved_model)
    if found:
        return value
    return config.MAX_THINKING_TOKENS


# A resolved id carrying this prefix is answered by the HTTP engine, not the CLI.
# The prefix is also a passthrough_prefix in models.json, so `openrouter/<vendor>/
# <model>` reaches the gateway unmapped and a new upstream model needs no code.
OPENROUTER_PREFIX = "openrouter/"


def engine_for(resolved_model: str) -> str:
    """Which engine answers a resolved model id: ``"openrouter"`` or ``"cli"``.

    The model map is the routing policy. Putting the engine in the id rather than
    in a second table means one hot-reloadable edit re-points a tier, and a client
    that asks for a model by name has already said which engine it wants.
    """
    return "openrouter" if (resolved_model or "").startswith(OPENROUTER_PREFIX) else "cli"


def claude_fallback(resolved_model: str) -> str:
    """The Claude model that answers when a constraint forces a non-CLI id back.

    A reroute (company data attached, a native document, the engine switched off)
    is not a statement about how hard the question is, so the fallback matches the
    tier the caller was already willing to accept rather than escalating.
    """
    table = _load().get("claude_fallback", {})
    return table.get(resolved_model) or _default_model(_load())


def is_text_only(resolved_model: str) -> bool:
    """True when the upstream model cannot accept images.

    Table-driven (``text_only`` in the model map) rather than hard-coded, so a
    later vision-capable tier is a config edit and not a release.
    """
    return resolved_model in _load().get("text_only", [])


def is_fast_model(resolved_model: str) -> bool:
    """True if a resolved --model belongs to the latency-sensitive fast tier.

    Reads the models file's ``fast_models`` list (hot-reloaded, like resolve_effort /
    resolve_max_thinking_tokens), defaulting to ``["haiku"]`` when absent. The engine
    uses this to pick the fast semaphore lane so interactive calls don't queue behind
    long heavy extraction jobs; every non-fast model uses the heavy lane.
    """
    fast = _load().get("fast_models", ["haiku"])
    return resolved_model in fast or model_tier(resolved_model) in fast


_EFFORT_VALUES = {"low", "medium", "high", "xhigh", "max"}


def parse_model_spec(requested: str) -> tuple[str, str | None]:
    """Split a client model string of the form ``name[:effort]`` into a resolved
    ``--model`` value and an optional per-request effort override.

    Lets a caller choose both per request (e.g. ``opus:high``, ``haiku:low``,
    ``sonnet``) with no gateway config change — the suffix is only treated as
    effort when it is a valid effort keyword, so model names that legitimately
    contain a colon are left untouched.
    """
    effort: str | None = None
    if requested:
        head, sep, tail = requested.rpartition(":")
        if sep and head and tail in _EFFORT_VALUES:
            requested, effort = head, tail
    return resolve_model(requested), effort


def list_model_ids() -> list[str]:
    """Advertised model ids: every alias key plus the canonical Claude aliases."""
    data = _load()
    ids = list(data.get("aliases", {}).keys())
    for extra in ("sonnet", "opus", "haiku"):
        if extra not in ids:
            ids.append(extra)
    return ids


def openai_models_payload() -> dict:
    return {
        "object": "list",
        "data": [
            {"id": i, "object": "model", "created": 0, "owned_by": "claude-gateway"}
            for i in list_model_ids()
        ],
    }


def gemini_models_payload() -> dict:
    return {
        "models": [
            {
                "name": f"models/{i}",
                "supportedGenerationMethods": ["generateContent", "streamGenerateContent"],
            }
            for i in list_model_ids()
        ],
    }
