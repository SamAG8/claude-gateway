# Context

## Glossary

**Claude Gateway** — the Python HTTP server that exposes Anthropic, OpenAI, and Gemini-compatible APIs and answers them with the local Claude CLI. A client written for any of the three services connects by changing only its `base_url`.

**Invocation** — a single stateless call to the `claude` CLI via subprocess (stream-json in, stream-json out). No conversation history is retained between invocations; multi-turn requests are replayed each call.

**Adapter** — a per-protocol module (`gateway/adapters/{anthropic,openai,gemini}.py`) that validates auth, translates the protocol's request into a Canonical Request, and supplies a Formatter to the Renderer. Each adapter holds only the protocol-specific translation and formatting; the event-driving and termination logic live once in the Renderer.

**Canonical Request / Canonical Event** — the internal contract (`gateway/canonical.py`) that every adapter speaks to the engine. The request carries model, system text, messages (text/image blocks), and stream flag; the engine yields a typed Canonical Event union — `Start` / `Delta` / `Stop` / `Error` — that the Renderer dispatches on. The engine never imports an adapter.

**Renderer** — the deep module (`gateway/protocol.py`) that drives the engine's Canonical Event stream once for every protocol, owning event ordering, termination, and the stream-vs-`collect` split. It crosses a single seam: the Formatter.

**Formatter** — the seam the Renderer crosses: a small, per-request, per-protocol Adapter (defined inside each `gateway/adapters/*` module) that renders each Canonical Event into that protocol's SSE chunks and builds its non-streaming body. Two+ Formatters make the seam real.

**Engine** — `gateway/engine.py`; the dispatcher. It selects which transport answers a request (`select_engine`), drives it, and drains it for non-streaming callers (`collect`). It never imports an adapter, and the adapters never learn which transport ran.

**CLI Engine** — `gateway/engines/cli.py`; the original and default transport. Builds the `claude` command line and stdin, spawns the subprocess in a Lane with a per-invocation timeout, parses the `stream-json` output, and yields Canonical Events. The only transport that can reach company data, because the MCP server is attached to the CLI.

**HTTP Engine** — `gateway/engines/openrouter.py`; the second transport. Streams from a third-party API for models whose resolved id carries the `openrouter/` prefix. It has no tool loop and bills real money per token, where the CLI Engine rides this machine's Claude login.

**Route Reason** — why a request is not being answered by the engine its model named (`mcp`, `document`, `image`, `openrouter-disabled`, `openrouter-<status>`), recorded on the Canonical Request and in the Usage Log. Null on the common path; a value is a constraint overriding the client's choice, and counting them is how we learn whether the routing is reaching the intended engine.

**Isolation Mode** — how the gateway neutralizes the machine's personal context so it behaves like a clean model API. `clean` (default): override the system prompt, load no settings/hooks (`--setting-sources ""`), disable tools (`--tools ""`), and run in a throwaway cwd — keeping the machine's subscription/OAuth login. `bare`: add `--bare` (requires `ANTHROPIC_API_KEY`).

**Model Map** — `models.json` (resolved by `gateway/models.py`, hot-reloaded by mtime); resolves a client's model string to a real `claude --model` value via passthrough (`claude-*`) → alias → default. Unknown models fall back to the default rather than erroring. Operational policy is family-aware: dated ids such as `claude-haiku-4-5-20251001` inherit the `haiku` fast-lane, effort, and thinking-budget settings.

**Lane** — `gateway/lanes.py`; a named semaphore plus one policy: wait at most `QUEUE_WAIT_MAX` seconds for a slot, then fail fast with a 503 rather than sit in an invisible queue. Three of them — `fast` (CLI, the latency-sensitive haiku tier, `MAX_CONCURRENT_FAST`), `heavy` (CLI, everything else, `MAX_CONCURRENT`), and `http` (the HTTP Engine, `MAX_CONCURRENT_HTTP`). `http` never shares with the CLI lanes: a subprocess costs CPU and RAM on this host and an HTTP stream costs a socket, so one capacity cannot describe both.

**Concurrency Cap** — the capacity of a Lane. Excess requests queue on its semaphore; they are rejected only when the queue wait runs out.

**API Key** — a shared secret presented in each protocol's native auth header (`x-api-key` / `Authorization: Bearer` / `x-goog-api-key` or `?key=`), compared constant-time against the configured key set (`API_KEY` plus optional comma-separated `API_KEYS`).

**Stream** — a Server-Sent Events response carrying that protocol's incremental events (Anthropic `message_*`/`content_block_*`, OpenAI `chat.completion.chunk` + `[DONE]`, Gemini partial `GenerateContentResponse`s) as Claude generates them.

**Timeout** — the maximum wall-clock seconds a single Invocation may run before the subprocess is killed and an error event is sent (`TIMEOUT`, default 120).

**Document / Image input** — images are passed inline to the CLI as base64 (native vision). PDFs are sent as a native `document` block on both surfaces, so Claude reads the pages with vision — a drawing's lines, a scan, a table's layout (`document_block` on Gemini, `pdf_block` on Anthropic). The Anthropic surface flattens a PDF to extracted text via pdfplumber (`pdf_to_text_block`) only when the caller sends `x-pdf-mode: text`, or when the PDF has more than `MAX_PDF_PAGES` pages, which the Messages API would refuse natively. Inbound base64 may be standard or URL-safe and is normalized to canonical standard base64 (`normalize_b64`); `MAX_FILE_SIZE` is enforced on decoded image bytes and `MAX_PDF_SIZE` on decoded PDF bytes.

**Usage Log** — per-invocation accounting with queue/spawn/stdin/first-event/first-text/total timings, token/cache counters, prompt-byte/history-message/media counts, model/lane, MCP flag, outcome, and reference cost. It never stores prompt/response content or credentials. `scripts/usage_report.py` aggregates P50/P95/P99 by model and MCP path; `scripts/benchmark_latency.py` measures deployed end-to-end TTFT/total percentiles.
