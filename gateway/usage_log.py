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

logger = logging.getLogger("claude-gateway.usage")

_dir_ready = False


def _ensure_dir(path: Path) -> None:
    global _dir_ready
    if not _dir_ready:
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        _dir_ready = True


def record(*, outcome: str, req, elapsed: float,
           input_tokens=None, output_tokens=None,
           cache_read=None, cache_creation=None,
           num_images: int = 0, num_docs: int = 0, media_bytes: int = 0) -> None:
    if not config.USAGE_LOG:
        return
    try:
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "surface": getattr(req, "surface", "") or "",
            "outcome": outcome,
            "requested_model": req.requested_model,
            "model": req.model,
            "elapsed_s": round(elapsed, 3),
            "input_tokens": input_tokens or 0,
            "output_tokens": output_tokens or 0,
            "cache_read": cache_read or 0,
            "cache_creation": cache_creation or 0,
            "num_images": num_images,
            "num_docs": num_docs,
            "media_bytes": media_bytes,
        }
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
