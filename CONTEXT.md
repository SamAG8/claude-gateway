# Context

## Glossary

**Claude Gateway** — the Python HTTP server that exposes Anthropic, OpenAI, and Gemini-compatible APIs and answers them with the local Claude CLI. A client written for any of the three services connects by changing only its `base_url`.

**Invocation** — a single stateless call to the `claude` CLI via subprocess (stream-json in, stream-json out). No conversation history is retained between invocations; multi-turn requests are replayed each call.

**Adapter** — a per-protocol module (`gateway/adapters/{anthropic,openai,gemini}.py`) that validates auth, translates the protocol's request into a Canonical Request, calls the engine, and formats the Canonical Event stream back into that protocol's exact response shape and native error envelope.

**Canonical Request / Canonical Event** — the internal contract (`gateway/canonical.py`) that every adapter speaks to the engine. The request carries model, system text, messages (text/image blocks), and stream flag; the engine yields a tagged event stream (`start` / `delta` / `stop` / `error`) that adapters render. The engine never imports an adapter.

**Engine** — `gateway/engine.py`; builds the `claude` command line and stdin, spawns the subprocess under a concurrency semaphore with a per-invocation timeout, parses the `stream-json` output, and yields Canonical Events. One code path serves streaming and non-streaming.

**Isolation Mode** — how the gateway neutralizes the machine's personal context so it behaves like a clean model API. `clean` (default): override the system prompt, load no settings/hooks (`--setting-sources ""`), disable tools (`--tools ""`), and run in a throwaway cwd — keeping the machine's subscription/OAuth login. `bare`: add `--bare` (requires `ANTHROPIC_API_KEY`).

**Model Map** — `models.json` (resolved by `gateway/models.py`, hot-reloaded by mtime); resolves a client's model string to a real `claude --model` value via passthrough (`claude-*`) → alias → default. Unknown models fall back to the default rather than erroring.

**Concurrency Cap** — the maximum number of simultaneous Invocations (`MAX_CONCURRENT`, default 5). Excess requests queue on the semaphore, they are not rejected.

**API Key** — a shared secret presented in each protocol's native auth header (`x-api-key` / `Authorization: Bearer` / `x-goog-api-key` or `?key=`), compared constant-time against the configured key set (`API_KEY` plus optional comma-separated `API_KEYS`).

**Stream** — a Server-Sent Events response carrying that protocol's incremental events (Anthropic `message_*`/`content_block_*`, OpenAI `chat.completion.chunk` + `[DONE]`, Gemini partial `GenerateContentResponse`s) as Claude generates them.

**Timeout** — the maximum wall-clock seconds a single Invocation may run before the subprocess is killed and an error event is sent (`TIMEOUT`, default 120).

**Document / Image input** — images are passed inline to the CLI as base64 (native vision); PDFs (Anthropic `document` blocks) are extracted to text via pdfplumber before the call. `MAX_FILE_SIZE` is enforced on decoded bytes.
