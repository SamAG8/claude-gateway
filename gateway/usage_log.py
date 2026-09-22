"""Best-effort structured usage log: one JSON object per invocation.

Appends to config.USAGE_LOG (JSONL) when set — in ADDITION to the human-readable
line the engine writes to journald. A no-op when USAGE_LOG is empty. Never raises
into the request path (a logging failure must not fail an invocation).

Aggregate it with scripts/usage_report.py.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from . import config, pricing
from .canonical import media_stats

logger = logging.getLogger("claude-gateway.usage")

# The per-invocation journald line keeps the ENGINE logger name and the
# `run_claude …` / `run_openrouter …` verb it has always had: operators grep for
# both, and renaming them to match this module's new home would break every saved
# query for a tidiness nobody asked for.
_engine_logger = logging.getLogger("claude-gateway.engine")

_dir_ready = False

_REASON_MAX = 200


def _short_reason(reason: str | None) -> str | None:
    """One-line, length-capped reason so a log line stays greppable."""
    if not reason:
        return None
    return " ".join(str(reason).split())[:_REASON_MAX] or None


def outcome(verb: str, result: str, req, elapsed: float,
            in_tok: int | None = None, out_tok: int | None = None,
            cache_read: int | None = None, cache_creation: int | None = None,
            lane: str | None = None, queue_wait_ms: int | None = None,
            spawn_ms: int | None = None,
            stdin_ms: int | None = None, first_event_ms: int | None = None,
            first_text_ms: int | None = None, total_ms: int | None = None,
            prompt_bytes: int = 0, history_messages: int = 0,
            mcp: bool = False,
            provider: str | None = None,
            cost_usd: float | None = None,
            reasoning_tokens: int | None = None,
            reason: str | None = None,
            level: int = logging.INFO) -> None:
    """One line per invocation so errors and durations are visible in journald,
    plus a structured JSONL record (when USAGE_LOG is set) for aggregation.

    Lives here rather than in an engine because both engines end the same way, and
    because it lets them depend on this module instead of on each other.

    ``reason`` is the upstream's own explanation of a non-success outcome. Log it:
    without it an operator sees only the word "error" and has to reproduce the
    request to find out what happened. That cost real time in the 2026-07-29
    ConstraAP incident — five ~200s `error` lines with no hint that the cause was
    `API Error: 529 Overloaded`, which was sitting in the 502 body all along.
    """
    num_images, num_docs, media_bytes = media_stats(req)
    _engine_logger.log(
        level,
        "%s %s surface=%s model=%s engine=%s route=%s lane=%s mcp=%s queue_ms=%s spawn_ms=%s "
        "stdin_ms=%s first_event_ms=%s first_text_ms=%s total_ms=%s "
        "elapsed=%.1fs in=%s out=%s cache_read=%s cache_write=%s imgs=%s docs=%s%s",
        verb, result, req.surface or "-", req.model,
        getattr(req, "engine", "cli"), getattr(req, "route_reason", None) or "-",
        lane or "-", mcp, queue_wait_ms,
        spawn_ms, stdin_ms, first_event_ms, first_text_ms, total_ms,
        elapsed, in_tok, out_tok,
        cache_read, cache_creation, num_images, num_docs,
        f" reason={_short_reason(reason)}" if reason else "")
    record(outcome=result, req=req, elapsed=elapsed,
           input_tokens=in_tok, output_tokens=out_tok,
           cache_read=cache_read, cache_creation=cache_creation,
           num_images=num_images, num_docs=num_docs, media_bytes=media_bytes,
           lane=lane, queue_wait_ms=queue_wait_ms, spawn_ms=spawn_ms,
           stdin_ms=stdin_ms, first_event_ms=first_event_ms,
           first_text_ms=first_text_ms, total_ms=total_ms,
           prompt_bytes=prompt_bytes, history_messages=history_messages,
           mcp=mcp, provider=provider, cost_usd=cost_usd,
           reasoning_tokens=reasoning_tokens,
           reason=_short_reason(reason))


def _ensure_dir(path: Path) -> None:
    global _dir_ready
    if not _dir_ready:
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        _dir_ready = True


def record(*, outcome: str, req, elapsed: float,
           input_tokens=None, output_tokens=None,
           cache_read=None, cache_creation=None,
           num_images: int = 0, num_docs: int = 0, media_bytes: int = 0,
           lane=None, queue_wait_ms=None, spawn_ms=None, stdin_ms=None,
           first_event_ms=None, first_text_ms=None, total_ms=None,
           prompt_bytes: int = 0, history_messages: int = 0,
           mcp: bool = False, provider=None, cost_usd=None,
           reasoning_tokens=None, reason=None) -> None:
    if not config.USAGE_LOG:
        return
    try:
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "surface": getattr(req, "surface", "") or "",
            "outcome": outcome,
            "requested_model": req.requested_model,
            "model": req.model,
            # Which engine answered, and why it is not the one the model asked for.
            # route_reason is null on the common path; a value means a constraint
            # overrode the client's choice, and counting those is how we learn
            # whether the routing is actually reaching the cheap engine.
            "engine": getattr(req, "engine", "cli"),
            "route_reason": getattr(req, "route_reason", None),
            "lane": lane,
            "introspect_ms": getattr(req, "introspect_ms", None),
            "queue_wait_ms": queue_wait_ms,
            "spawn_ms": spawn_ms,
            "stdin_ms": stdin_ms,
            "first_event_ms": first_event_ms,
            "first_text_ms": first_text_ms,
            "total_ms": total_ms,
            "prompt_bytes": prompt_bytes,
            "history_messages": history_messages,
            "mcp": bool(mcp),
            "elapsed_s": round(elapsed, 3),
            "input_tokens": input_tokens or 0,
            "output_tokens": output_tokens or 0,
            "cache_read": cache_read or 0,
            "cache_creation": cache_creation or 0,
            "num_images": num_images,
            "num_docs": num_docs,
            "media_bytes": media_bytes,
        }
        # Only present on non-success outcomes, so success records stay unchanged.
        if reason:
            rec["reason"] = reason
        # Only the HTTP engine has these, and only it bills real money. est_cost_usd
        # below stays a REFERENCE figure for every record; cost_usd is what was
        # actually charged, which is the number the subscription-vs-cash split needs.
        if provider:
            rec["provider"] = provider
        if cost_usd is not None:
            rec["cost_usd"] = cost_usd
        if reasoning_tokens is not None:
            rec["reasoning_tokens"] = reasoning_tokens
        rec["est_cost_usd"] = pricing.estimate_cost_usd(
            req.model, rec["input_tokens"], rec["output_tokens"],
            rec["cache_read"], rec["cache_creation"],
        )
        path = Path(config.USAGE_LOG)
        _ensure_dir(path)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:  # never break a request over logging
        logger.warning("usage_log write failed: %s", e)
