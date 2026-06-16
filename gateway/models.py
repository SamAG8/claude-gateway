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
