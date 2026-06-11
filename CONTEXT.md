# Context

## Glossary

**Claude Gateway** — the Python HTTP server that accepts prompts from external clients and returns Claude's responses. Runs on the same machine as the Claude CLI.

**Invocation** — a single stateless call to `claude -p "<prompt>"` via subprocess. No conversation history is retained between invocations.

**Concurrency Cap** — the maximum number of simultaneous Invocations allowed. Defaults to 5, configurable via environment variable. Excess requests are queued, not rejected.

**API Key** — a shared secret passed by clients in the `X-API-Key` request header. Validated server-side against an environment variable. Required on all requests.

**Stream** — the HTTP response to a client: a Server-Sent Events stream of text tokens produced by Claude as they are generated, rather than a single response returned after completion.

**Timeout** — the maximum wall-clock seconds a single Invocation may run before the subprocess is killed and an error event is sent to the client. Defaults to 120 seconds, configurable via environment variable.

**Document Analysis** — a variant of Invocation that processes an uploaded file rather than a raw text prompt. PDFs are handled by extracting their text content via pdfplumber before passing it to Claude. Images (PNG, JPG, GIF, WebP) are sent directly to Claude as base64-encoded content via the CLI's stream-json input mode, leveraging Claude's native vision capability.

**Upload Directory** — a temporary directory on disk (default: `/tmp/gateway-uploads`) where incoming files are staged for analysis. Each request gets an isolated UUID subdirectory that is deleted as soon as its stream completes.
