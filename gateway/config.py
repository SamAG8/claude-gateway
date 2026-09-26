"""Environment-driven configuration for the gateway.

All runtime knobs are read once at import. See README / .env.example for docs.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _load_api_keys() -> set[str]:
    """Collect accepted shared secrets from API_KEY and comma-separated API_KEYS."""
    keys: set[str] = set()
    single = os.getenv("API_KEY", "").strip()
    if single:
        keys.add(single)
    for part in os.getenv("API_KEYS", "").split(","):
        part = part.strip()
        if part:
            keys.add(part)
    return keys


API_KEYS = _load_api_keys()

# "clean" (default): system-prompt override + no settings/hooks, keeps subscription auth.
# "bare": adds --bare (skips hooks/LSP/memory/CLAUDE.md) but forces ANTHROPIC_API_KEY auth.
ISOLATION_MODE = os.getenv("ISOLATION_MODE", "clean").strip().lower()

MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "5"))  # heavy lane (opus/sonnet) capacity
# Dedicated fast-lane capacity so interactive fast-tier (haiku) calls don't queue
# behind long-running heavy extraction jobs. See models.is_fast_model + engine lanes.
MAX_CONCURRENT_FAST = int(os.getenv("MAX_CONCURRENT_FAST", "3"))
# The HTTP engine's own lane. Sized far higher than the CLI lanes because an
# in-flight OpenRouter stream costs a socket, not a subprocess with its own CPU
# and RAM on this host. See gateway/lanes.py for why it never shares with them.
MAX_CONCURRENT_HTTP = int(os.getenv("MAX_CONCURRENT_HTTP", "20"))
# Max seconds a request may wait to acquire its lane's semaphore slot before the
# gateway gives up and returns a fast 503 ("saturated, retry") instead of letting
# the client burn its whole timeout budget in an invisible queue. See engine.
QUEUE_WAIT_MAX = float(os.getenv("QUEUE_WAIT_MAX", "10"))
TIMEOUT = int(os.getenv("TIMEOUT", "120"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(10 * 1024 * 1024)))  # 10 MB
# A PDF sent natively is read page by page with Claude's vision, and the Messages API
# takes one of up to 32 MB and 100 pages. Construction drawing sets routinely pass
# 10 MB, so PDFs get their own ceiling rather than the image one. Past the page
# limit a PDF is flattened to text instead of being refused (content.pdf_block).
MAX_PDF_SIZE = int(os.getenv("MAX_PDF_SIZE", str(32 * 1024 * 1024)))  # 32 MB
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "100"))

# Max bytes for a single stream-json line read from the `claude` CLI (the asyncio
# StreamReader limit). The default 64 KiB is far too small: with --verbose the CLI
# echoes the user message — inline base64 media included — as one NDJSON line, so a
# request with an image overruns and the read fails (issue #11). The engine
# additionally scales this up to the actual stdin payload per request.
STREAM_LIMIT = int(os.getenv("STREAM_LIMIT", str(32 * 1024 * 1024)))  # 32 MiB

# Editable model map (hot-reloaded by mtime). DEFAULT_MODEL overrides its "default" when set.
MODELS_FILE = os.getenv("MODELS_FILE", "models.json")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "").strip()

# Structured per-invocation usage log (one JSON line per call): tokens, cache
# hits, media counts, elapsed, reference cost. Empty = disabled (the engine still
# writes its human-readable line to journald). Aggregate with scripts/usage_report.py.
USAGE_LOG = os.getenv("USAGE_LOG", "").strip()

# Reasoning effort passed to `claude --effort` (low|medium|high|xhigh|max). Empty =
# use the CLI default. Set EFFORT=high (or xhigh) to give Opus 4.8 a larger thinking
# budget — e.g. for ConstraBid bid extraction (Claude Gateway V1).
EFFORT = os.getenv("EFFORT", "").strip()

# Global override for the `claude` CLI's MAX_THINKING_TOKENS env var. When set, it
# applies to every resolved model lacking a per-model entry in the models file's
# ``max_thinking_tokens`` map. Default None = do NOT inject the env var, so the CLI
# uses its own default thinking budget (opus/sonnet extraction keeps thinking).
# MAX_THINKING_TOKENS=0 fully disables extended thinking (the CLI maps 0 to
# {type:"disabled"}, not clamped to 1024). Per-model map entries win; see models.py.
MAX_THINKING_TOKENS = int(os.getenv("MAX_THINKING_TOKENS")) if os.getenv("MAX_THINKING_TOKENS") else None

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# --- OpenRouter engine (the second transport) ------------------------------
# Unlike the CLI engine, which rides this machine's Claude login, OpenRouter bills
# real money per token. It answers only models whose resolved id carries the
# ``openrouter/`` prefix (see models.engine_for), so it is inert until a client
# asks for one. Unset the key to switch the whole tier back to Claude without a
# deploy — engine.select_engine reroutes and says so in route_reason.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip().rstrip("/")
# Idle-read timeout: the longest gap between two SSE chunks before we give up.
# The upstream sends keepalive comment lines, so a silent connection really is dead.
OPENROUTER_TIMEOUT = float(os.getenv("OPENROUTER_TIMEOUT", "60"))
OPENROUTER_CONNECT_TIMEOUT = float(os.getenv("OPENROUTER_CONNECT_TIMEOUT", "5"))
# How the upstream picks among its providers for a model. "latency" is the point
# of this engine; set "" to leave the choice to OpenRouter's default (price).
OPENROUTER_PROVIDER_SORT = os.getenv("OPENROUTER_PROVIDER_SORT", "latency").strip()
# Fall back to the Claude model in claude_fallback when OpenRouter fails BEFORE the
# first chunk. Never after: a half-streamed answer cannot be replaced. Set 0 if you
# would rather see the failure than quietly reload the subscription during an
# upstream outage — either way it is visible as route_reason in the usage log.
OPENROUTER_FALLBACK = os.getenv("OPENROUTER_FALLBACK", "1").strip() not in ("0", "false", "no")
# Open one connection at startup so the first real turn does not pay the TLS
# handshake. Failure is ignored; it is a warm-up, not a health check.
OPENROUTER_WARM = os.getenv("OPENROUTER_WARM", "1").strip() not in ("0", "false", "no")
# Optional attribution shown on the OpenRouter dashboard.
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "").strip()
OPENROUTER_REFERER = os.getenv("OPENROUTER_REFERER", "").strip()


def openrouter_enabled() -> bool:
    return bool(OPENROUTER_API_KEY)

# --- Audio transcription (Nimbus Meeting Minutes) -----------------------------
# One narrow audio path: POST /v1/audio/transcriptions takes a meeting recording,
# ffmpeg reduces it to small mono speech audio, and an audio-capable OpenRouter
# model writes the transcript. Nothing else in the gateway handles audio.
TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "google/gemini-2.5-flash").strip()
TRANSCRIBE_MAX_UPLOAD = int(os.getenv("TRANSCRIBE_MAX_UPLOAD", str(600 * 1024 * 1024)))
TRANSCRIBE_MAX_SECONDS = int(os.getenv("TRANSCRIBE_MAX_SECONDS", str(90 * 60)))
# Recordings longer than this are cut into segments of this length and
# transcribed one segment at a time, so no single answer runs into max_tokens.
TRANSCRIBE_SEGMENT_SECONDS = int(os.getenv("TRANSCRIBE_SEGMENT_SECONDS", str(30 * 60)))
TRANSCRIBE_TIMEOUT = float(os.getenv("TRANSCRIBE_TIMEOUT", "300"))
TRANSCRIBE_MAX_TOKENS = int(os.getenv("TRANSCRIBE_MAX_TOKENS", "32000"))
# Comma-separated ConstraAP org ids allowed to transcribe. Empty = any valid PAT.
TRANSCRIBE_ORG_IDS = {v.strip() for v in os.getenv("TRANSCRIBE_ORG_IDS", "").split(",") if v.strip()}
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg").strip()
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe").strip()


def transcribe_enabled() -> bool:
    import shutil
    return openrouter_enabled() and bool(shutil.which(FFMPEG_BIN)) and bool(shutil.which(FFPROBE_BIN))

# --- MCP connector (per-user company data) ---------------------------------
# When MCP_SERVER_URL is set, a request carrying a per-user token (the
# `x-mcp-token` header) runs the CLI with that MCP server attached, scoped to its
# tools only. This lets a client (e.g. Nimbus) give Claude live access to
# ConstraAP data without a native tool API — the CLI calls the MCP tools and
# returns final text. The token is per-request (per user); the URL + tool scope
# are gateway config. Built-in tools stay disabled, so isolation is preserved.
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "").strip()
MCP_SERVER_NAME = os.getenv("MCP_SERVER_NAME", "constraap").strip()


def mcp_enabled() -> bool:
    return bool(MCP_SERVER_URL)


# --- Per-user PAT auth (login is the single credential) --------------------
# When TOKEN_INTROSPECT_URL is set, a request may authenticate with the user's own
# ConstraAP PAT (cap_…) instead of the shared API_KEY: the gateway POSTs the token
# to this endpoint and accepts it iff {"active": true}. The validated PAT is then
# reused as the per-user MCP token, so company data is scoped to that user for
# free. INTROSPECT_SECRET (if set) is sent as x-introspect-secret so the endpoint
# isn't an open oracle. The shared API_KEY still works as a dev/local fallback.
TOKEN_INTROSPECT_URL = os.getenv("TOKEN_INTROSPECT_URL", "").strip()
INTROSPECT_SECRET = os.getenv("INTROSPECT_SECRET", "").strip()
INTROSPECT_CACHE_TTL = int(os.getenv("INTROSPECT_CACHE_TTL", "60"))
INTROSPECT_TIMEOUT = int(os.getenv("INTROSPECT_TIMEOUT", "8"))


def pat_auth_enabled() -> bool:
    return bool(TOKEN_INTROSPECT_URL)


# Throwaway working dir for the CLI subprocess so no CLAUDE.md / project files leak in.
CLEAN_CWD = Path(os.getenv("GATEWAY_CLEAN_CWD", "/tmp/claude-gateway-clean"))

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
