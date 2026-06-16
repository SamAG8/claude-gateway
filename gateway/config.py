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

# Editable model map (hot-reloaded by mtime). DEFAULT_MODEL overrides its "default" when set.
MODELS_FILE = os.getenv("MODELS_FILE", "models.json")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "").strip()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Throwaway working dir for the CLI subprocess so no CLAUDE.md / project files leak in.
CLEAN_CWD = Path(os.getenv("GATEWAY_CLEAN_CWD", "/tmp/claude-gateway-clean"))

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
