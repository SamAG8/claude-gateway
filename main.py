import asyncio
import base64
import json
import os
import shutil
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
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


async def _stream_subprocess(args: list):
    """Run claude with given args and stream text output as SSE events."""
    async with semaphore:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

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
        ["claude", "-p", prompt, "--output-format", "text", "--tools", "none"]
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
