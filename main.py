import asyncio
import base64
import json
import os
import shutil
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv()

API_KEY = os.getenv("API_KEY", "")
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "5"))
TIMEOUT = int(os.getenv("TIMEOUT", "120"))
PORT = int(os.getenv("PORT", "8000"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/gateway-uploads"))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(10 * 1024 * 1024)))  # 10 MB

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

EXTENSION_TO_MEDIA_TYPE = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
IMAGE_EXTENSIONS = set(EXTENSION_TO_MEDIA_TYPE)
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | {".pdf"}

DEFAULT_ANALYZE_INSTRUCTIONS = (
    "Analyze this document. If it is an invoice, extract: vendor name, invoice number, "
    "invoice date, due date, line items with descriptions and amounts, subtotal, tax, and total. "
    "If it is another type of document, describe its contents and extract key information."
)

semaphore = asyncio.Semaphore(MAX_CONCURRENT)

app = FastAPI(title="Claude Gateway")


class ChatRequest(BaseModel):
    prompt: str


def verify_api_key(x_api_key: str = Header(...)):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY not configured on server")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


async def _stream_subprocess(args: list, stdin_data: bytes | None = None):
    """Run claude with given args and stream text output as SSE events."""
    async with semaphore:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE if stdin_data is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if stdin_data is not None:
            process.stdin.write(stdin_data)
            await process.stdin.drain()
            process.stdin.close()

        loop = asyncio.get_event_loop()
        start = loop.time()

        try:
            while True:
                remaining = TIMEOUT - (loop.time() - start)
                if remaining <= 0:
                    process.kill()
                    await process.wait()
                    yield f"data: {json.dumps({'status': 'error', 'answer': 'timeout'})}\n\n"
                    return
                try:
                    chunk = await asyncio.wait_for(
                        process.stdout.read(1024),
                        timeout=remaining,
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    yield f"data: {json.dumps({'status': 'error', 'answer': 'timeout'})}\n\n"
                    return

                if not chunk:
                    break

                yield f"data: {json.dumps({'status': 'streaming', 'answer': chunk.decode('utf-8', errors='replace')})}\n\n"

            await process.wait()
            yield f"data: {json.dumps({'status': 'done', 'answer': None})}\n\n"

        except Exception as e:
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass
            yield f"data: {json.dumps({'status': 'error', 'answer': str(e)})}\n\n"


async def _stream_image_via_json(stdin_msg: bytes):
    """Send an image to Claude via stream-json stdin and stream SSE events back."""
    async with semaphore:
        process = await asyncio.create_subprocess_exec(
            "claude",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",
            "--tools", "none",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        process.stdin.write(stdin_msg)
        await process.stdin.drain()
        process.stdin.close()

        loop = asyncio.get_event_loop()
        start = loop.time()

        try:
            while True:
                remaining = TIMEOUT - (loop.time() - start)
                if remaining <= 0:
                    process.kill()
                    await process.wait()
                    yield f"data: {json.dumps({'status': 'error', 'answer': 'timeout'})}\n\n"
                    return

                try:
                    line = await asyncio.wait_for(process.stdout.readline(), timeout=remaining)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    yield f"data: {json.dumps({'status': 'error', 'answer': 'timeout'})}\n\n"
                    return

                if not line:
                    break

                try:
                    event = json.loads(line)
                    etype = event.get("type")

                    if etype == "assistant":
                        for block in event.get("message", {}).get("content", []):
                            if block.get("type") == "text":
                                yield f"data: {json.dumps({'status': 'streaming', 'answer': block['text']})}\n\n"

                    elif etype == "result":
                        if event.get("subtype") == "success":
                            yield f"data: {json.dumps({'status': 'done', 'answer': None})}\n\n"
                        else:
                            yield f"data: {json.dumps({'status': 'error', 'answer': event.get('result', 'analysis failed')})}\n\n"
                except json.JSONDecodeError:
                    pass

            await process.wait()

        except Exception as e:
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass
            yield f"data: {json.dumps({'status': 'error', 'answer': str(e)})}\n\n"


# Swap this function for an SSH-based invoker when remote Claude support is added.
async def invoke_claude(prompt: str):
    async for event in _stream_subprocess(
        ["claude", "-p", prompt, "--output-format", "text", "--tools", "none"]
    ):
        yield event


async def invoke_claude_with_file(file_path: Path, instructions: str):
    ext = file_path.suffix.lower()
    try:
        if ext == ".pdf":
            async for event in _analyze_pdf(file_path, instructions):
                yield event
        else:
            async for event in _analyze_image(file_path, instructions, EXTENSION_TO_MEDIA_TYPE[ext]):
                yield event
    finally:
        shutil.rmtree(file_path.parent, ignore_errors=True)


async def _analyze_pdf(file_path: Path, instructions: str):
    import pdfplumber

    try:
        with pdfplumber.open(str(file_path)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        text = "\n\n".join(pages).strip()
    except Exception as e:
        yield f"data: {json.dumps({'status': 'error', 'answer': f'PDF extraction failed: {e}'})}\n\n"
        return

    if not text:
        yield f"data: {json.dumps({'status': 'error', 'answer': 'No text could be extracted from the PDF'})}\n\n"
        return

    prompt = f"{instructions}\n\nDocument content:\n{text}"
    async for event in _stream_subprocess(
        ["claude", "--input-format", "text", "--output-format", "text", "--tools", "none"],
        stdin_data=prompt.encode(),
    ):
        yield event


async def _analyze_image(file_path: Path, instructions: str, media_type: str):
    try:
        with open(file_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
    except Exception as e:
        yield f"data: {json.dumps({'status': 'error', 'answer': f'File read failed: {e}'})}\n\n"
        return

    stdin_msg = json.dumps({
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": b64},
                },
                {"type": "text", "text": instructions},
            ],
        },
    }).encode()

    async for event in _stream_image_via_json(stdin_msg):
        yield event


_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Claude Gateway</title>
<script src="https://cdn.jsdelivr.net/npm/marked@9/marked.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#f0f0f0;color:#1a1a1a;padding:2rem}
.wrap{max-width:780px;margin:0 auto}
h1{font-size:1.4rem;font-weight:700;margin-bottom:1.5rem}
.card{background:#fff;border-radius:10px;padding:1.5rem;margin-bottom:1rem;box-shadow:0 1px 4px rgba(0,0,0,.08)}
label{display:block;font-size:.78rem;font-weight:600;text-transform:uppercase;letter-spacing:.05em;color:#555;margin-bottom:.4rem}
input,textarea{width:100%;border:1px solid #ddd;border-radius:6px;padding:.6rem .8rem;font-size:.9rem;font-family:inherit;outline:none;transition:border .15s}
input:focus,textarea:focus{border-color:#555}
textarea{resize:vertical;min-height:90px}
.tabs{display:flex;gap:.4rem;margin-bottom:1.2rem}
.tab{padding:.45rem 1rem;border:1px solid #ddd;border-radius:6px;cursor:pointer;background:#fff;font-size:.85rem;font-weight:500;transition:all .15s}
.tab.active{background:#1a1a1a;color:#fff;border-color:#1a1a1a}
.panel{display:none}.panel.active{display:block}
.field{margin-bottom:1rem}
.hint{font-size:.75rem;color:#888;margin-top:.3rem}
button[type=submit]{background:#1a1a1a;color:#fff;border:none;border-radius:6px;padding:.65rem 1.4rem;font-size:.9rem;font-weight:500;cursor:pointer;margin-top:.5rem;transition:background .15s}
button[type=submit]:hover{background:#333}
button[type=submit]:disabled{background:#aaa;cursor:not-allowed}
/* Response card */
.resp-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:.85rem}
.badge{font-size:.72rem;font-weight:600;padding:.2rem .55rem;border-radius:4px;text-transform:uppercase;letter-spacing:.05em}
.badge-thinking{background:#f3e5f5;color:#6a1b9a;animation:pulse 1.2s ease-in-out infinite}
.badge-streaming{background:#e3f2fd;color:#1565c0;animation:pulse .9s ease-in-out infinite}
.badge-done{background:#e8f5e9;color:#1b5e20}
.badge-error{background:#fce4ec;color:#880e4f}
@keyframes pulse{0%,100%{opacity:.5}50%{opacity:1}}
/* Prose response box */
.resp-box{
  background:#fafafa;border:1px solid #e8e8e8;border-radius:8px;
  padding:1.25rem 1.5rem;min-height:80px;
  font-family:system-ui,-apple-system,sans-serif;font-size:.95rem;
  line-height:1.75;color:#1a1a1a;
  transition:opacity .3s ease;
}
.resp-box.waiting{color:#aaa;font-style:italic;font-size:.9rem}
.resp-box h1{font-size:1.25rem;font-weight:700;margin:1.1rem 0 .5rem;padding-bottom:.35rem;border-bottom:1px solid #eee;color:#111}
.resp-box h2{font-size:1.05rem;font-weight:700;margin:.95rem 0 .4rem;color:#111}
.resp-box h3{font-size:.97rem;font-weight:600;margin:.8rem 0 .3rem;color:#333}
.resp-box h1:first-child,.resp-box h2:first-child,.resp-box h3:first-child{margin-top:0}
.resp-box p{margin:.55rem 0}
.resp-box p:first-child{margin-top:0}
.resp-box p:last-child{margin-bottom:0}
.resp-box ul,.resp-box ol{margin:.5rem 0 .5rem 1.6rem}
.resp-box li{margin:.25rem 0}
.resp-box li>p{margin:.15rem 0}
.resp-box strong{font-weight:700}
.resp-box em{font-style:italic}
.resp-box code{font-family:ui-monospace,Menlo,monospace;font-size:.82rem;background:#efefef;padding:.1em .38em;border-radius:3px;color:#c7254e}
.resp-box pre{background:#f4f4f5;border:1px solid #e4e4e7;border-radius:6px;padding:.9rem 1.1rem;overflow-x:auto;margin:.75rem 0}
.resp-box pre code{background:none;padding:0;color:#1a1a1a;font-size:.83rem}
.resp-box blockquote{border-left:3px solid #d1d5db;margin:.7rem 0;padding:.35rem .9rem;color:#6b7280}
.resp-box table{border-collapse:collapse;width:100%;margin:.8rem 0;font-size:.9rem}
.resp-box th{background:#f3f4f6;font-weight:600;padding:.45rem .85rem;border:1px solid #d1d5db;text-align:left}
.resp-box td{padding:.4rem .85rem;border:1px solid #d1d5db}
.resp-box tr:nth-child(even) td{background:#f9fafb}
.resp-box hr{border:none;border-top:1px solid #e5e7eb;margin:1rem 0}
/* Copy button */
.copy-row{display:none;justify-content:flex-end;margin-top:.75rem}
.copy-btn{display:inline-flex;align-items:center;gap:.35rem;background:#fff;border:1px solid #d1d5db;border-radius:6px;padding:.4rem .85rem;font-size:.82rem;font-weight:500;color:#555;cursor:pointer;transition:all .15s;font-family:inherit}
.copy-btn:hover{background:#f3f4f6;border-color:#9ca3af;color:#1a1a1a}
.copy-btn.copied{border-color:#86efac;color:#166534;background:#f0fdf4}
</style>
</head>
<body>
<div class="wrap">
  <h1>Claude Gateway</h1>

  <div class="card">
    <div class="field">
      <label for="apikey">API Key</label>
      <input type="password" id="apikey" placeholder="Paste your X-API-Key here">
      <div class="hint">Stored in your browser — never sent anywhere except this server.</div>
    </div>
  </div>

  <div class="card">
    <div class="tabs">
      <button class="tab active" data-tab="chat">Chat</button>
      <button class="tab" data-tab="analyze">Analyze Document</button>
    </div>

    <div id="panel-chat" class="panel active">
      <form id="form-chat">
        <div class="field">
          <label for="prompt">Prompt</label>
          <textarea id="prompt" placeholder="Ask Claude anything…"></textarea>
        </div>
        <button type="submit">Send</button>
      </form>
    </div>

    <div id="panel-analyze" class="panel">
      <form id="form-analyze">
        <div class="field">
          <label for="file">File</label>
          <input type="file" id="file" accept=".png,.jpg,.jpeg,.gif,.webp,.pdf">
          <div class="hint">PNG · JPG · GIF · WebP · PDF — max 10 MB</div>
        </div>
        <div class="field">
          <label for="instructions">Instructions <span style="font-weight:400;text-transform:none">(optional)</span></label>
          <textarea id="instructions" rows="2" placeholder="Leave blank to use default invoice extraction…"></textarea>
        </div>
        <button type="submit">Analyze</button>
      </form>
    </div>
  </div>

  <div class="card" id="resp-card" style="display:none">
    <div class="resp-header">
      <span class="badge" id="badge"></span>
    </div>
    <div class="resp-box" id="resp"></div>
    <div class="copy-row" id="copy-row">
      <button class="copy-btn" id="copy-btn" type="button" onclick="copyText()">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
        </svg>
        Copy response
      </button>
    </div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
let lastRawText = '';

marked.setOptions({ breaks: true, gfm: true });

const keyEl = $('apikey');
keyEl.value = localStorage.getItem('cgw_key') || '';
keyEl.addEventListener('input', () => localStorage.setItem('cgw_key', keyEl.value));

document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(x => x.classList.remove('active'));
  t.classList.add('active');
  $('panel-' + t.dataset.tab).classList.add('active');
}));

function setBadge(state, label) {
  const b = $('badge');
  b.textContent = label || state;
  b.className = 'badge badge-' + state;
}

async function animateText(text) {
  const el = $('resp');
  el.classList.remove('waiting');
  el.style.opacity = '1';
  // Phase 1: animate as plain text so it feels like it's streaming in
  el.textContent = '';
  const step = Math.max(3, Math.ceil(text.length / 70));
  for (let i = 0; i < text.length; i += step) {
    el.textContent = text.slice(0, i + step);
    await new Promise(r => setTimeout(r, 16));
  }
  // Phase 2: cross-fade into rendered markdown
  el.style.opacity = '0';
  el.innerHTML = marked.parse(text);
  await new Promise(r => setTimeout(r, 30));
  el.style.opacity = '1';
}

async function copyText() {
  const btn = $('copy-btn');
  try {
    await navigator.clipboard.writeText(lastRawText);
  } catch {
    // Fallback for browsers that block clipboard API
    const ta = Object.assign(document.createElement('textarea'), {value: lastRawText});
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
  }
  btn.classList.add('copied');
  btn.querySelector('span') && (btn.querySelector('span').textContent = 'Copied!');
  const svg = btn.querySelector('svg');
  const prevHTML = btn.innerHTML;
  btn.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg> Copied!';
  setTimeout(() => {
    btn.innerHTML = prevHTML;
    btn.classList.remove('copied');
  }, 2200);
}

async function stream(url, init) {
  const respEl = $('resp');
  $('resp-card').style.display = 'block';
  $('copy-row').style.display = 'none';
  respEl.className = 'resp-box waiting';
  respEl.style.opacity = '1';
  respEl.textContent = 'Thinking…';
  setBadge('thinking', 'Thinking…');

  let fullText = '';
  let hasError = false;

  try {
    const res = await fetch(url, init);
    if (!res.ok) {
      const err = await res.json().catch(() => ({detail: res.statusText}));
      setBadge('error', 'Error');
      respEl.className = 'resp-box';
      respEl.textContent = err.detail || res.statusText;
      return;
    }

    setBadge('streaming', 'Receiving…');
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      const lines = buf.split('\\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const ev = JSON.parse(line.slice(6));
          if (ev.status === 'streaming' && ev.answer) {
            fullText += ev.answer;
          } else if (ev.status === 'error') {
            hasError = true;
            if (ev.answer) fullText += '\\n\\n> ⚠ ' + ev.answer;
          }
        } catch {}
      }
    }

    lastRawText = fullText;
    setBadge('streaming', 'Rendering…');
    await animateText(fullText);
    setBadge(hasError ? 'error' : 'done', hasError ? 'Error' : 'Done');
    $('copy-row').style.display = 'flex';

  } catch (err) {
    setBadge('error', 'Error');
    respEl.className = 'resp-box';
    respEl.textContent = err.message;
  }
}

$('form-chat').addEventListener('submit', async e => {
  e.preventDefault();
  const key = keyEl.value.trim(), prompt = $('prompt').value.trim();
  if (!key) return alert('Enter your API key');
  if (!prompt) return alert('Enter a prompt');
  const btn = e.target.querySelector('button');
  btn.disabled = true; btn.textContent = 'Sending…';
  await stream('/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json', 'X-API-Key': key},
    body: JSON.stringify({prompt}),
  });
  btn.disabled = false; btn.textContent = 'Send';
});

$('form-analyze').addEventListener('submit', async e => {
  e.preventDefault();
  const key = keyEl.value.trim(), file = $('file').files[0];
  if (!key) return alert('Enter your API key');
  if (!file) return alert('Select a file');
  const btn = e.target.querySelector('button');
  btn.disabled = true; btn.textContent = 'Analyzing…';
  const fd = new FormData();
  fd.append('file', file);
  const instr = $('instructions').value.trim();
  if (instr) fd.append('instructions', instr);
  await stream('/analyze', {
    method: 'POST',
    headers: {'X-API-Key': key},
    body: fd,
  });
  btn.disabled = false; btn.textContent = 'Analyze';
});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def ui():
    return _UI_HTML


@app.post("/chat")
async def chat(request: ChatRequest, _: None = Depends(verify_api_key)):
    return StreamingResponse(
        invoke_claude(request.prompt),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    instructions: str = Form(default=DEFAULT_ANALYZE_INSTRUCTIONS),
    _: None = Depends(verify_api_key),
):
    ext = Path(file.filename or "upload").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    request_dir = UPLOAD_DIR / str(uuid.uuid4())
    request_dir.mkdir(parents=True, exist_ok=True)
    file_path = request_dir / f"document{ext}"

    try:
        content = await file.read(MAX_FILE_SIZE + 1)
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds {MAX_FILE_SIZE // (1024 * 1024)} MB limit",
            )
        file_path.write_bytes(content)
    except HTTPException:
        shutil.rmtree(request_dir, ignore_errors=True)
        raise

    return StreamingResponse(
        invoke_claude_with_file(file_path, instructions),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
