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
    for prefix in data.get("passthrough_prefixes", []):
        if requested.startswith(prefix):
            return requested
    aliases = data.get("aliases", {})
    if requested in aliases:
        return aliases[requested]
    return _default_model(data)


def model_tier(resolved_model: str) -> str:
    """Return the stable haiku/sonnet/opus family for a resolved CLI model id.

    Clients commonly send dated ids such as ``claude-haiku-4-5-20251001``.
    Those ids pass through to the CLI, but operational policy (fast lane,
    effort, thinking budget) must still match the family-level ``haiku`` entry
    in models.json. Unknown ids remain exact so custom model keys still work.
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


def is_fast_model(resolved_model: str) -> bool:
    """True if a resolved --model belongs to the latency-sensitive fast tier.

    Reads the models file's ``fast_models`` list (hot-reloaded, like resolve_effort /
    resolve_max_thinking_tokens), defaulting to ``["haiku"]`` when absent. The engine
    uses this to pick the fast semaphore lane so interactive calls don't queue behind
    long heavy extraction jobs; every non-fast model uses the heavy lane.
    """
    fast = _load().get("fast_models", ["haiku"])
    return resolved_model in fast or model_tier(resolved_model) in fast


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
