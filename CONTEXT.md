# Context

## Glossary

**Claude Gateway** — the Python HTTP server that exposes Anthropic, OpenAI, and Gemini-compatible APIs and answers them with the local Claude CLI. A client written for any of the three services connects by changing only its `base_url`.

**Invocation** — a single stateless call to the `claude` CLI via subprocess (stream-json in, stream-json out). No conversation history is retained between invocations; multi-turn requests are replayed each call.

**Adapter** — a per-protocol module (`gateway/adapters/{anthropic,openai,gemini}.py`) that validates auth, translates the protocol's request into a Canonical Request, and supplies a Formatter to the Renderer. Each adapter holds only the protocol-specific translation and formatting; the event-driving and termination logic live once in the Renderer.

**Canonical Request / Canonical Event** — the internal contract (`gateway/canonical.py`) that every adapter speaks to the engine. The request carries model, system text, messages (text/image blocks), and stream flag; the engine yields a typed Canonical Event union — `Start` / `Delta` / `Stop` / `Error` — that the Renderer dispatches on. The engine never imports an adapter.

**Renderer** — the deep module (`gateway/protocol.py`) that drives the engine's Canonical Event stream once for every protocol, owning event ordering, termination, and the stream-vs-`collect` split. It crosses a single seam: the Formatter.

**Formatter** — the seam the Renderer crosses: a small, per-request, per-protocol Adapter (defined inside each `gateway/adapters/*` module) that renders each Canonical Event into that protocol's SSE chunks and builds its non-streaming body. Two+ Formatters make the seam real.

**Engine** — `gateway/engine.py`; builds the `claude` command line and stdin, spawns the subprocess under a concurrency semaphore with a per-invocation timeout, parses the `stream-json` output, and yields Canonical Events. One code path serves streaming and non-streaming.

**Isolation Mode** — how the gateway neutralizes the machine's personal context so it behaves like a clean model API. `clean` (default): override the system prompt, load no settings/hooks (`--setting-sources ""`), disable tools (`--tools ""`), and run in a throwaway cwd — keeping the machine's subscription/OAuth login. `bare`: add `--bare` (requires `ANTHROPIC_API_KEY`).

**Model Map** — `models.json` (resolved by `gateway/models.py`, hot-reloaded by mtime); resolves a client's model string to a real `claude --model` value via passthrough (`claude-*`) → alias → default. Unknown models fall back to the default rather than erroring.

**Concurrency Cap** — the maximum number of simultaneous Invocations (`MAX_CONCURRENT`, default 5). Excess requests queue on the semaphore, they are not rejected.

**API Key** — a shared secret presented in each protocol's native auth header (`x-api-key` / `Authorization: Bearer` / `x-goog-api-key` or `?key=`), compared constant-time against the configured key set (`API_KEY` plus optional comma-separated `API_KEYS`).

**Stream** — a Server-Sent Events response carrying that protocol's incremental events (Anthropic `message_*`/`content_block_*`, OpenAI `chat.completion.chunk` + `[DONE]`, Gemini partial `GenerateContentResponse`s) as Claude generates them.

**Timeout** — the maximum wall-clock seconds a single Invocation may run before the subprocess is killed and an error event is sent (`TIMEOUT`, default 120).

**Document / Image input** — images are passed inline to the CLI as base64 (native vision). PDFs are handled two ways: the Gemini surface sends `application/pdf` inline data as a native `document` block (Claude reads it with vision, preserving layout/handwriting/highlights via `document_block`); the Anthropic surface flattens `document` blocks to extracted text via pdfplumber (`pdf_to_text_block`). Inbound base64 may be standard or URL-safe and is normalized to canonical standard base64 (`normalize_b64`); `MAX_FILE_SIZE` is enforced on decoded bytes.
