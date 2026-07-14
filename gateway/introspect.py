"""Validate per-user ConstraAP PATs (cap_… tokens) at the gateway door.

The gateway's original door lock is a shared static secret (see errors.key_is_valid).
This adds a second, per-user path: a request may instead present the user's own
ConstraAP PAT, which we validate by calling ConstraAP's /mcp-tokens/introspect
endpoint. That makes the user's login token the single credential — it both opens
the model and (reused as the MCP token) scopes company data to that user.

Results are cached briefly so we don't round-trip per message. We FAIL CLOSED: if
introspection is unconfigured or unreachable, the token is treated as invalid.
No new dependency — a plain urllib POST on a worker thread.
"""
import asyncio
import json
import time
import urllib.request

from . import config

# token -> (expires_at_monotonic, active). Small user base; pruned opportunistically.
_cache: dict[str, tuple[float, bool]] = {}
_MAX_CACHE = 4096


def _prune(now: float) -> None:
    if len(_cache) <= _MAX_CACHE:
        return
    for k in [k for k, (exp, _) in _cache.items() if exp <= now]:
        _cache.pop(k, None)


def _introspect_sync(token: str) -> bool:
    body = json.dumps({"token": token}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if config.INTROSPECT_SECRET:
        headers["x-introspect-secret"] = config.INTROSPECT_SECRET
    req = urllib.request.Request(
        config.TOKEN_INTROSPECT_URL, data=body, headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=config.INTROSPECT_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return bool(data.get("active"))
    except Exception:
        return False  # fail closed: unreachable / bad response => not authorized


async def token_is_active(token: str | None) -> bool:
    """True if `token` is a live ConstraAP PAT. Cached; fail-closed."""
    if not token or not config.TOKEN_INTROSPECT_URL:
        return False
    now = time.monotonic()
    hit = _cache.get(token)
    if hit and hit[0] > now:
        return hit[1]
    active = await asyncio.to_thread(_introspect_sync, token)
    # Cache positives for the full TTL; negatives briefly so a revoke propagates fast
    # and a transient outage doesn't lock a good token out for long.
    ttl = config.INTROSPECT_CACHE_TTL if active else min(config.INTROSPECT_CACHE_TTL, 15)
    _cache[token] = (now + ttl, active)
    _prune(now)
    return active
