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

MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "5"))
TIMEOUT = int(os.getenv("TIMEOUT", "120"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(10 * 1024 * 1024)))  # 10 MB

# Max bytes for a single stream-json line read from the `claude` CLI (the asyncio
# StreamReader limit). The default 64 KiB is far too small: with --verbose the CLI
# echoes the user message — inline base64 media included — as one NDJSON line, so a
# request with an image overruns and the read fails (issue #11). The engine
# additionally scales this up to the actual stdin payload per request.
STREAM_LIMIT = int(os.getenv("STREAM_LIMIT", str(32 * 1024 * 1024)))  # 32 MiB

# Editable model map (hot-reloaded by mtime). DEFAULT_MODEL overrides its "default" when set.
MODELS_FILE = os.getenv("MODELS_FILE", "models.json")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "").strip()

# Reasoning effort passed to `claude --effort` (low|medium|high|xhigh|max). Empty =
# use the CLI default. Set EFFORT=high (or xhigh) to give Opus 4.8 a larger thinking
# budget — e.g. for ConstraBid bid extraction (Claude Gateway V1).
EFFORT = os.getenv("EFFORT", "").strip()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

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
