"""claude-gateway FastAPI app.

A multi-protocol, drop-in model API backed by the local Claude CLI. This module
wires the app together: CORS, an unauthenticated /health probe, and a startup
config log. Protocol routers (Anthropic / OpenAI / Gemini) are mounted here as
they land.
"""
import logging
import subprocess
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gateway import config
from gateway.adapters import anthropic, gemini, openai
from gateway.engines.cli import ensure_clean_cwd

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("claude-gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_clean_cwd()
    log.info(
        "claude-gateway up: isolation=%s max_concurrent=%s timeout=%ss api_keys=%d models_file=%s",
        config.ISOLATION_MODE, config.MAX_CONCURRENT, config.TIMEOUT,
        len(config.API_KEYS), config.MODELS_FILE,
    )
    if not config.API_KEYS:
        log.warning("No API_KEY/API_KEYS configured — all authenticated requests will be rejected.")
    yield


app = FastAPI(title="Claude Gateway", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(anthropic.router)
app.include_router(openai.router)
app.include_router(gemini.router)


@lru_cache(maxsize=1)
def deployed_revision() -> str:
    """The build this process is running, or "unknown".

    A REVISION file first, because THE DEPLOY THAT MATTERS HAS NO GIT. The
    serving instance is packaged by ConstraAP's
    `scripts/deploy-to-production.sh --gateway`, which tars this directory and
    rsyncs it with `--exclude .git` — so a `git rev-parse` here would answer
    "unknown" on the only box anybody cares about. The packaging step writes
    the file; this reads it.

    Git is the fallback, for a server bootstrapped by scripts/deploy.sh (which
    does `git reset --hard`, making HEAD exactly what is serving) and for a
    developer running uvicorn out of a checkout.

    Cached: a subprocess per health check would turn a liveness probe into a
    fork bomb under a load balancer.
    """
    here = Path(__file__).resolve().parent

    stamped = here / "REVISION"
    try:
        value = stamped.read_text(encoding="utf-8").strip()
        if value:
            return value
    except OSError:
        pass

    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=here, capture_output=True, text=True, timeout=2, check=True,
        ).stdout.strip() or "unknown"
    except Exception:
        # No REVISION file, no git. Not knowing is a fine answer; failing a
        # health check over it is not.
        return "unknown"


@app.get("/health")
async def health():
    """Liveness, and WHICH BUILD is answering.

    This used to return `{"status": "ok"}` and nothing else, which made a
    reasonable question — "is the fix I merged actually running?" — impossible
    to answer without SSH. A capability that exists in the repository and not in
    production looks exactly like a capability that was never written, and the
    x-mcp-token sentinel sat in that gap for three weeks.

    `mcp` is here for the same reason: whether company data can be attached at
    all is the single most consequential piece of this gateway's configuration,
    and it was equally invisible. Both are facts about the deployment, not
    secrets — no token, no URL, no key.
    """
    return {
        "status": "ok",
        "revision": deployed_revision(),
        "mcp": config.mcp_enabled(),
        "pat_auth": config.pat_auth_enabled(),
    }


if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT)
