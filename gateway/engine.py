"""Core engine: build the `claude` invocation, spawn it, and yield CanonicalEvents.

One code path serves every adapter and both streaming and non-streaming modes.
Streaming adapters consume the events live; non-streaming adapters drain them.
"""
import asyncio
import json
from typing import AsyncIterator

from . import config
from .canonical import (
    CanonicalEvent,
    CanonicalRequest,
    Delta,
    Error,
    Result,
    Start,
    Stop,
    map_stop_reason,
)

_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    # Created lazily so it binds to the running event loop.
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(config.MAX_CONCURRENT)
    return _semaphore


def ensure_clean_cwd() -> None:
    """Create the throwaway cwd used for every invocation (no CLAUDE.md leaks in)."""
    config.CLEAN_CWD.mkdir(parents=True, exist_ok=True)


def build_argv(req: CanonicalRequest) -> list[str]:
    """Assemble the contamination-neutralized `claude` command line."""
    argv = [
        "claude", "-p",
        "--model", req.model,
        "--input-format", "stream-json",
        "--output-format", "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--no-session-persistence",
        "--tools", "",            # empty string = disable ALL tools (NOT the word "none")
        "--setting-sources", "",  # do not load user/project/local settings (where hooks live)
        "--system-prompt", req.system or config.DEFAULT_SYSTEM_PROMPT,
    ]
    if config.ISOLATION_MODE == "bare":
        argv.append("--bare")
    return argv


def _image_to_cli(block: dict) -> dict:
    """Map an image content block to its CLI `image` base64 source form."""
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": block["media_type"], "data": block["data"]},
    }


def _document_to_cli(block: dict) -> dict:
    """Map a document content block to its CLI `document` base64 source form."""
    return {
        "type": "document",
        "source": {"type": "base64", "media_type": block["media_type"], "data": block["data"]},
    }


def _media_to_cli(block: dict) -> dict | None:
    """Map an image/document content block to its CLI source form (None otherwise)."""
    if block.get("type") == "image":
        return _image_to_cli(block)
    if block.get("type") == "document":
        return _document_to_cli(block)
    return None


def _text_of(blocks: list[dict]) -> str:
    return " ".join(b.get("text", "") for b in blocks if b.get("type") == "text" and b.get("text"))


def _flatten_turn(blocks: list[dict]) -> str:
    """Render a history turn as text; images collapse to a placeholder (best-effort)."""
    parts = []
    for b in blocks:
        if b.get("type") == "text" and b.get("text"):
            parts.append(b["text"])
        elif b.get("type") == "image":
            parts.append("[image omitted]")
        elif b.get("type") == "document":
            parts.append("[document omitted]")
    return " ".join(parts)


def build_stdin(req: CanonicalRequest) -> bytes:
    """Build the stream-json user message sent on stdin.

    Single user turn -> sent directly (images/documents preserved as native blocks).
    Multi-turn       -> prior turns flattened into a transcript prepended to the
                        final user text; the final turn's images/documents are preserved.
    """
    messages = req.messages or []
    final = messages[-1] if messages else None
    history = messages[:-1]

    if not history:
        content = []
        for b in (final.blocks if final else []):
            if b.get("type") == "text":
                content.append({"type": "text", "text": b.get("text", "")})
            elif (media := _media_to_cli(b)) is not None:
                content.append(media)
    else:
        lines = ["[conversation so far]"]
        for m in history:
            label = "User" if m.role == "user" else "Assistant"
            lines.append(f"{label}: {_flatten_turn(m.blocks)}")
        lines.append("[end]")
        lines.append("Now respond to the final user message:")
        lines.append("")
        transcript = "\n".join(lines)
        final_text = _text_of(final.blocks) if final else ""
        combined = transcript + (("\n" + final_text) if final_text else "")
        content = [{"type": "text", "text": combined}]
        for b in (final.blocks if final else []):
            if (media := _media_to_cli(b)) is not None:
                content.append(media)

    msg = {"type": "user", "message": {"role": "user", "content": content}}
    return (json.dumps(msg) + "\n").encode()


async def run_claude(req: CanonicalRequest) -> AsyncIterator[CanonicalEvent]:
    """Spawn `claude` for one stateless invocation and yield CanonicalEvents."""
    if config.ISOLATION_MODE == "bare" and not config.ANTHROPIC_API_KEY:
        yield Error(500, "ISOLATION_MODE=bare requires ANTHROPIC_API_KEY in the environment")
        return

    ensure_clean_cwd()
    argv = build_argv(req)
    stdin_data = build_stdin(req)

    async with _get_semaphore():
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(config.CLEAN_CWD),
            )
        except FileNotFoundError:
            yield Error(500, "claude CLI not found on PATH")
            return

        try:
            process.stdin.write(stdin_data)
            await process.stdin.drain()
            process.stdin.close()
        except (BrokenPipeError, ConnectionResetError):
            pass

        loop = asyncio.get_event_loop()
        start = loop.time()
        started = False
        cap_stop = None
        cap_out = None
        cap_in = None

        try:
            while True:
                remaining = config.TIMEOUT - (loop.time() - start)
                if remaining <= 0:
                    process.kill()
                    await process.wait()
                    yield Error(504, "upstream timeout")
                    return
                try:
                    line = await asyncio.wait_for(process.stdout.readline(), timeout=remaining)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    yield Error(504, "upstream timeout")
                    return

                if not line:
                    break

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                otype = obj.get("type")
                if otype == "stream_event":
                    ev = obj.get("event", {})
                    etype = ev.get("type")
                    if etype == "message_start":
                        msg = ev.get("message", {})
                        usage = msg.get("usage", {})
                        cap_in = usage.get("input_tokens")
                        started = True
                        yield Start(model=msg.get("model"), input_tokens=usage.get("input_tokens", 0))
                    elif etype == "content_block_delta":
                        delta = ev.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield Delta(text=delta.get("text", ""))
                    elif etype == "message_delta":
                        cap_stop = ev.get("delta", {}).get("stop_reason", cap_stop)
                        u = ev.get("usage", {})
                        if u.get("output_tokens") is not None:
                            cap_out = u["output_tokens"]
                elif otype == "result":
                    if obj.get("is_error") or obj.get("subtype") != "success":
                        yield Error(502, obj.get("result") or "upstream error")
                        await process.wait()
                        return
                    usage = obj.get("usage", {})
                    yield Stop(
                        stop_reason=map_stop_reason(obj.get("stop_reason") or cap_stop),
                        output_tokens=usage.get("output_tokens", cap_out or 0),
                        input_tokens=usage.get("input_tokens", cap_in or 0),
                    )
                    await process.wait()
                    return
                # ignore: system, assistant, rate_limit_event, hook/status lines

            # stdout closed without a result line
            await process.wait()
            if not started:
                err = b""
                try:
                    err = await process.stderr.read()
                except Exception:
                    pass
                msg = err.decode("utf-8", errors="replace").strip()[:500]
                yield Error(502, msg or "no output from claude")
            else:
                yield Stop(stop_reason=map_stop_reason(cap_stop),
                           output_tokens=cap_out or 0, input_tokens=cap_in or 0)
        except Exception as e:  # noqa: BLE001 - surface any spawn/read failure as an error event
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass
            yield Error(500, str(e))


async def collect(req: CanonicalRequest) -> Result:
    """Drain run_claude into a single non-streaming Result for adapters to format."""
    text_parts: list[str] = []
    model = req.requested_model
    stop_reason = "end_turn"
    input_tokens = 0
    output_tokens = 0
    error: Error | None = None
    async for ev in run_claude(req):
        if isinstance(ev, Start):
            model = ev.model or model
            input_tokens = ev.input_tokens
        elif isinstance(ev, Delta):
            text_parts.append(ev.text)
        elif isinstance(ev, Stop):
            stop_reason = ev.stop_reason
            output_tokens = ev.output_tokens
            input_tokens = ev.input_tokens
        elif isinstance(ev, Error):
            error = ev
            break
    return Result(
        text="".join(text_parts),
        model=model,
        stop_reason=stop_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        error=error,
    )
