import asyncio
import json
import os

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

API_KEY = os.getenv("API_KEY", "")
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "5"))
TIMEOUT = int(os.getenv("TIMEOUT", "120"))
PORT = int(os.getenv("PORT", "8000"))

semaphore = asyncio.Semaphore(MAX_CONCURRENT)

app = FastAPI(title="Claude Gateway")


class ChatRequest(BaseModel):
    prompt: str


def verify_api_key(x_api_key: str = Header(...)):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY not configured on server")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# Swap this function for an SSH-based invoker when remote Claude support is added.
async def invoke_claude(prompt: str):
    async with semaphore:
        process = await asyncio.create_subprocess_exec(
            "claude", "-p", prompt, "--output-format", "text",
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


@app.post("/chat")
async def chat(request: ChatRequest, _: None = Depends(verify_api_key)):
    return StreamingResponse(
        invoke_claude(request.prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
