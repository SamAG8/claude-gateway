# claude-gateway

A multi-protocol, **drop-in model API** backed by the local Claude CLI. Point any
Anthropic, OpenAI, or Google Gemini SDK at this server by changing only the
`base_url` — the gateway speaks all three wire protocols and runs `claude` locally
behind the scenes.

```
Anthropic SDK ─► POST /v1/messages ─────────────┐
OpenAI SDK    ─► POST /v1/chat/completions ──────┼─► canonical request ─► claude CLI
Gemini SDK    ─► POST /v1beta/models/*:generate* ┘        (isolated)      └─► canonical events
                                                                              │
                                              each adapter ◄── formats that protocol's response/SSE
```

Each surface accepts **text + images**, supports **streaming and non-streaming**,
returns **real token usage**, and emits responses byte-shaped like the real upstream.

## Drop-in usage

```python
from anthropic import Anthropic
Anthropic(base_url="http://host:8000", api_key="GATEWAY_KEY").messages.create(
    model="claude-sonnet-4-6", max_tokens=100,
    messages=[{"role": "user", "content": "Say hi"}])

from openai import OpenAI
OpenAI(base_url="http://host:8000/v1", api_key="GATEWAY_KEY").chat.completions.create(
    model="gpt-4o", messages=[{"role": "user", "content": "Say hi"}])

import google.generativeai as genai   # client_options.api_endpoint -> http://host:8000
```

## Endpoints

| Surface | Endpoint(s) | Auth header |
|---|---|---|
| **Anthropic Messages** | `POST /v1/messages` | `x-api-key` (or `Authorization: Bearer`) |
| **OpenAI Chat Completions** | `POST /v1/chat/completions`, `GET /v1/models` | `Authorization: Bearer` |
| **Google Gemini** | `POST /v1beta/models/{model}:generateContent`, `:streamGenerateContent`, `GET /v1beta/models` | `x-goog-api-key` or `?key=` |
| Health | `GET /health` (no auth) | — |

## How it works

One shared core, three thin adapters:

- **Adapters** (`gateway/adapters/*`) translate each protocol's request into a
  **canonical request** and format the canonical event stream back into that
  protocol's response (JSON or SSE) and native error envelope.
- **The engine** (`gateway/engine.py`) builds one **isolated** `claude` invocation
  and parses its `stream-json` output into canonical events. The same code path
  serves streaming and non-streaming on every surface.

### Isolation (clean model behavior)

By default the gateway strips the machine's personal context so it behaves like a
clean model API, not your coding assistant. Every call passes `--system-prompt`
(client system message, or `"You are a helpful assistant."`), `--setting-sources ""`
(no user/project/local settings or `SessionStart` hooks), `--tools ""` (all tools
off), and runs in a throwaway working directory (no `CLAUDE.md`). Set
`ISOLATION_MODE=bare` to instead use `--bare` (requires `ANTHROPIC_API_KEY`); the
default `clean` keeps the machine's existing subscription/OAuth login.

### Model mapping

The client's model string is resolved to a real `claude --model` value via
`models.json` (editable, hot-reloaded by mtime):

1. Anything matching a `passthrough_prefixes` entry (default `claude-`) is passed
   straight through — so `claude-sonnet-4-6`, `sonnet`, `opus`, `haiku` work and
   stay current as Claude's aliases track the latest models.
2. Otherwise a known alias (e.g. `gpt-4o → sonnet`, `gemini-1.5-pro → opus`) maps to a tier.
3. Otherwise it falls back to the default. **Unknown models never error.**

## Setup

**Requirements:** Python 3.10+, the [Claude CLI](https://claude.ai/code) installed and authenticated.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # set a strong API_KEY
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `API_KEY` | — | Required shared secret. `API_KEYS` (comma-separated) adds more accepted keys. |
| `ISOLATION_MODE` | `clean` | `clean` (system-prompt override + no settings; keeps subscription auth) or `bare` (`--bare`, requires `ANTHROPIC_API_KEY`). |
| `MAX_CONCURRENT` | `5` | Max simultaneous CLI invocations (semaphore). |
| `TIMEOUT` | `120` | Per-invocation seconds before the subprocess is killed (→ 504). |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Bind address / port (read by `python3 main.py`). |
| `MAX_FILE_SIZE` | `10485760` | Max decoded image/document bytes (→ 413). |
| `MODELS_FILE` | `models.json` | Model-map path (hot-reloaded by mtime). |
| `DEFAULT_MODEL` | — | Overrides the map's `default` when set. |

## Verification (curl)

```bash
K=your-api-key
# OpenAI non-stream
curl -s localhost:8000/v1/chat/completions -H "Authorization: Bearer $K" \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Say hi"}]}'
# Anthropic non-stream
curl -s localhost:8000/v1/messages -H "x-api-key: $K" -H 'Content-Type: application/json' \
  -d '{"model":"claude-sonnet-4-6","max_tokens":100,"messages":[{"role":"user","content":"Say hi"}]}'
# Gemini stream (SSE)
curl -N "localhost:8000/v1beta/models/gemini-1.5-pro:streamGenerateContent?alt=sse" \
  -H "x-goog-api-key: $K" -H 'Content-Type: application/json' \
  -d '{"contents":[{"role":"user","parts":[{"text":"count to 3"}]}]}'
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest                 # mocked engine — no CLI calls
RUN_LIVE=1 pytest      # also runs the live contamination smoke test (needs claude)
```

## Known limitations (honesty)

The CLI does not expose sampling controls, so `temperature` / `top_p` / `top_k` /
`stop` / `n>1` / `logprobs` are **accepted but ignored**, and `max_tokens` is not
strictly enforced. Tool / function calling is accepted but ignored (no error).
Multi-turn history is replayed as a flattened transcript, and images in prior turns
are dropped to `[image omitted]` (the final turn's images are sent natively). Usage
`prompt_tokens` reflects the CLI's accounting.

## Security

- Constant-time API-key comparison; missing/invalid key → that protocol's 401 envelope.
- Tools disabled on every call — prompt injection cannot read files or run shell.
- `MAX_FILE_SIZE` enforced on decoded media before spawning; images sent inline (no disk).
- Permissive CORS so browser SDKs work; `/health` stays unauthenticated.

## Roadmap

- [ ] Tool / function calling across all three protocols
- [ ] Real multi-turn session reuse (`--session-id`) to preserve history images
- [ ] Per-client rate limiting
