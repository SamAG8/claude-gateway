"""claude-gateway FastAPI app.

A multi-protocol, drop-in model API backed by the local Claude CLI. This module
wires the app together: CORS, an unauthenticated /health probe, and a startup
config log. Protocol routers (Anthropic / OpenAI / Gemini) are mounted here as
they land.
"""
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gateway import config
from gateway.adapters import anthropic, gemini, openai
from gateway.engine import ensure_clean_cwd

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


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT)
