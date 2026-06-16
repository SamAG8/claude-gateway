# claude-gateway

A lightweight Python HTTP gateway that accepts prompts and documents from external clients and streams responses from Claude CLI via Server-Sent Events (SSE).

## How it works

```
Client → POST /chat   → FastAPI → subprocess: claude -p          → SSE stream → Client
Client → POST /analyze → FastAPI → pdfplumber / base64 → claude  → SSE stream → Client
```

Text prompts invoke `claude -p` directly. Document analysis extracts PDF text via pdfplumber, or sends images as base64 via the Claude CLI's stream-json input format. All responses stream back as SSE events.

## Setup

**Requirements:** Python 3.10+, [Claude CLI](https://claude.ai/code) installed and authenticated.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and set a strong API_KEY
```

## Configuration

All config lives in `.env`:

| Variable         | Default                  | Description                                      |
|------------------|--------------------------|--------------------------------------------------|
| `API_KEY`        | —                        | Required. Secret passed in `X-API-Key` header.  |
| `HOST`           | `0.0.0.0`                | Address the server binds to.                     |
| `PORT`           | `8000`                   | Port the server listens on.                      |
| `MAX_CONCURRENT` | `5`                      | Max simultaneous Claude invocations.             |
| `TIMEOUT`        | `120`                    | Seconds before a hung invocation is killed.      |
| `UPLOAD_DIR`     | `/tmp/gateway-uploads`   | Temp directory for file uploads.                 |
| `MAX_FILE_SIZE`  | `10485760`               | Max upload size in bytes (default 10 MB).        |

### Deployment (systemd)

`claude-gateway.service` reads `HOST` and `PORT` from the `.env` file at runtime
(via `EnvironmentFile`). The user, group, and install path are set by these
variables in `.env.example` and **must match** the corresponding directives in
the unit file, since systemd cannot expand `${VAR}` in `User=`, `Group=`,
`WorkingDirectory=`, or `EnvironmentFile=`:

| Variable        | Default                    | Unit directive it must match     |
|-----------------|----------------------------|----------------------------------|
| `SERVICE_USER`  | `gateway`                  | `User=`                          |
| `SERVICE_GROUP` | `gateway`                  | `Group=`                         |
| `INSTALL_DIR`   | `/home/gateway/claude-gateway` | `WorkingDirectory=` / `EnvironmentFile=` |

## Running

```bash
# Honors HOST/PORT from .env:
python3 main.py
# ...or run uvicorn directly:
uvicorn main:app --host "$HOST" --port "$PORT"
```

## API

### `POST /chat`

**Headers:**
```
X-API-Key: your-api-key
Content-Type: application/json
```

**Body:**
```json
{ "prompt": "your question here" }
```

**Response** (SSE stream):
```
data: {"status": "streaming", "answer": "Hello"}
data: {"status": "streaming", "answer": " there!"}
data: {"status": "done", "answer": null}
```

On error:
```
data: {"status": "error", "answer": "timeout"}
```

**Example:**
```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"prompt": "Explain async/await in one sentence"}'
```

---

### `POST /analyze`

Analyze an image or PDF document. Claude reads the file directly (images via multimodal vision, PDFs via text extraction).

**Headers:**
```
X-API-Key: your-api-key
```

**Form fields:**
| Field          | Required | Description                                          |
|----------------|----------|------------------------------------------------------|
| `file`         | Yes      | File upload. Accepted: `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.pdf` |
| `instructions` | No       | What to extract or analyze (defaults to invoice extraction) |

**Response** (SSE stream, same format as `/chat`):
```
data: {"status": "streaming", "answer": "Invoice #12345\nVendor: Acme Corp\nTotal: $500.00"}
data: {"status": "done", "answer": null}
```

**Example:**
```bash
curl -N -X POST http://localhost:8000/analyze \
  -H "X-API-Key: your-api-key" \
  -F "file=@invoice.pdf" \
  -F "instructions=Extract vendor name, invoice number, and total amount."
```

---

### `GET /health`

Returns `{"status": "ok"}`. No auth required.

## Security

- All tool access is disabled for text prompts (`--tools none`), preventing prompt injection from reading files or executing shell commands.
- Image analysis uses Claude's built-in vision (no file system access at all — the image is sent as base64 in the request payload).
- PDF text is extracted by pdfplumber in Python before being passed to Claude, with tools disabled.
- Each file upload is isolated in a per-request UUID subdirectory under `UPLOAD_DIR` and deleted immediately after the stream completes.

## Scalability notes

- **Concurrency:** The semaphore (`MAX_CONCURRENT`) prevents resource exhaustion under burst traffic. Tune it to your Claude CLI rate limit tier.
- **Timeout:** Each invocation is killed after `TIMEOUT` seconds, ensuring stuck processes release their semaphore slot.
- **Horizontal scaling:** Each instance manages its own semaphore. Put a load balancer in front to scale across multiple instances.

## Roadmap

- [ ] SSH-based remote invocation (swap `invoke_claude` in `main.py`)
- [ ] Multi-turn conversation sessions
- [ ] Per-client rate limiting
