# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`claude-gateway` is a multi-protocol, drop-in model API: a FastAPI server that simultaneously speaks the **Anthropic Messages**, **OpenAI Chat Completions**, and **Google Gemini** wire protocols, answering each by shelling out to the **local `claude` CLI**. Any SDK for those three services connects by changing only its `base_url`. Each surface supports text + images, streaming + non-streaming, real token usage, and byte-exact response/error shapes.

## Commands

```bash
# Setup
pip install -r requirements.txt          # runtime
pip install -r requirements-dev.txt      # + pytest, pytest-asyncio, httpx

# Run the server (honors HOST/PORT from .env)
python3 main.py
# or: uvicorn main:app --host 0.0.0.0 --port 8000

# Tests (asyncio_mode=auto, configured in pytest.ini)
pytest                                    # mocked engine — no CLI calls, fast
pytest tests/test_adapters.py            # one file
pytest tests/test_engine.py::test_run_claude_yields_canonical_events  # one test
RUN_LIVE=1 pytest tests/test_live_smoke.py   # hits the real claude CLI (costs tokens)
```

There is no build step and no linter configured.

## Architecture: one core, three adapters

The cardinal rule is **one shared engine, three thin adapters** — never duplicate CLI logic per protocol. Data flows:

```
adapter (protocol request → CanonicalRequest) → engine.run_claude() → CanonicalEvent stream → adapter (→ protocol response/SSE)
```

- **`gateway/canonical.py`** — the internal contract every adapter speaks to the core: `CanonicalRequest` / `CanonicalMessage` and a tagged `CanonicalEvent` dict stream (`start` / `delta` / `stop` / `error`). The engine **never imports an adapter**; adapters only ever hand the engine a `CanonicalRequest`.
- **`gateway/engine.py`** — `run_claude(req)` builds the `claude` argv + stdin, spawns the subprocess under a concurrency semaphore with a per-invocation timeout, parses the `stream-json` JSONL output, and yields `CanonicalEvent`s. `collect(req)` drains that same generator into one object for non-streaming callers. **This one code path serves both streaming and non-streaming on all three surfaces.**
- **`gateway/adapters/{anthropic,openai,gemini}.py`** — each is a FastAPI `APIRouter` that (1) validates that protocol's auth, (2) translates the request into a `CanonicalRequest`, (3) calls `run_claude`/`collect`, (4) formats the events into the protocol's exact JSON/SSE and native error envelope. Wired into the app in `main.py`.
- **`gateway/models.py` + `models.json`** — `resolve_model(requested)` maps a client model string to a real `claude --model` value: `passthrough_prefixes` (e.g. `claude-`) pass straight through, known aliases map to a tier, everything else falls back to the default. **Unknown models must never error** — they fall back. `models.json` is hot-reloaded by mtime.
- **`gateway/errors.py`** — constant-time multi-key auth (`key_is_valid` over `config.API_KEYS`) and per-protocol error envelope builders. **Each adapter must return its own protocol's native error shape**, not a generic one.
- **`gateway/content.py`** — decoding/validation of inbound media into canonical blocks: `MAX_FILE_SIZE` enforcement on decoded base64, PDF→text via pdfplumber, OpenAI data-URI parsing.

## Critical invariants

- **Contamination neutralization.** The engine must keep the gateway behaving like a clean model API, not the machine's coding assistant. Every invocation passes `--system-prompt` (client system or a default), `--setting-sources ""` (no user/project/local settings or `SessionStart` hooks), `--tools ""` (the empty string — **not** the word `none`, which the CLI parses as a tool name), `--no-session-persistence`, and runs in a throwaway `cwd` with no `CLAUDE.md`. `ISOLATION_MODE=bare` swaps to `--bare` (which then requires `ANTHROPIC_API_KEY`). The live smoke test asserts this: a trivial prompt returns small `input_tokens` with no leaked memory.
- **Parsing the CLI stream.** Consume `stream_event.event` payloads (they are 1:1 with Anthropic's wire events): `message_start`→`start`, `content_block_delta`(`text_delta`)→`delta`, `message_delta`→capture stop_reason/output_tokens, final `result`→`stop` (or `error` if `is_error`/`subtype!=success`). Ignore `system`, `assistant`, `rate_limit_event`, and hook lines.
- **Stateless multi-turn.** The CLI call is stateless; multi-turn requests flatten prior turns into a transcript prepended to the final user message. Only the **final** turn's images are sent natively — history images become `[image omitted]`.
- **Accept-but-ignore.** `temperature`/`top_p`/`top_k`/`stop`/`max_tokens`/tools are accepted and never error, but the CLI cannot enforce them. Don't invent CLI flags for them.

## Testing approach

Adapter and engine tests **mock, never call the real CLI**:
- `tests/conftest.py` provides `fake_claude` (monkeypatches `asyncio.create_subprocess_exec` with a canned stream-json transcript — for engine tests) and `mock_engine` (monkeypatches `engine.run_claude` to yield canonical events and capture the produced `CanonicalRequest` — for adapter tests, with an httpx ASGI `client` fixture).
- When adding a protocol feature, assert the **exact** event/`data:` sequence and field names (e.g. OpenAI's `[DONE]` sentinel, Anthropic's `message_start..message_stop` order, Gemini's final partial carrying `finishReason`+`usageMetadata`).

## Conventions

- Config is module-level constants in `gateway/config.py`, read once from env at import (see `.env.example`). Auth accepts `API_KEY` plus optional comma-separated `API_KEYS`.
- `CONTEXT.md` holds the project glossary (Adapter, Canonical Request/Event, Engine, Isolation Mode, Model Map) — keep it in sync when these concepts change.
- Scope is deliberately bounded: no tool/function calling, embeddings, audio, image-gen, batch, or multi-tenant key management (see README "Known limitations" and the issue's non-goals).
