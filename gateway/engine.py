"""Core engine: build the `claude` invocation, spawn it, and yield CanonicalEvents.

One code path serves every adapter and both streaming and non-streaming modes.
Streaming adapters consume the events live; non-streaming adapters drain them.
"""
import asyncio
import json
import logging
import os
from typing import AsyncIterator

from . import config, models, usage_log
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

logger = logging.getLogger("claude-gateway.engine")

_semaphore: asyncio.Semaphore | None = None


def _log_outcome(outcome: str, req: CanonicalRequest, elapsed: float,
                 in_tok: int | None = None, out_tok: int | None = None,
                 cache_read: int | None = None, cache_creation: int | None = None,
                 num_images: int = 0, num_docs: int = 0, media_bytes: int = 0,
                 level: int = logging.INFO) -> None:
    """One line per invocation so errors and durations are visible in journald,
    plus a structured JSONL record (when USAGE_LOG is set) for aggregation."""
    logger.log(level,
               "run_claude %s surface=%s model=%s elapsed=%.1fs in=%s out=%s "
               "cache_read=%s cache_write=%s imgs=%s docs=%s",
               outcome, req.surface or "-", req.model, elapsed, in_tok, out_tok,
               cache_read, cache_creation, num_images, num_docs)
    usage_log.record(outcome=outcome, req=req, elapsed=elapsed,
                     input_tokens=in_tok, output_tokens=out_tok,
                     cache_read=cache_read, cache_creation=cache_creation,
                     num_images=num_images, num_docs=num_docs, media_bytes=media_bytes)


def _get_semaphore() -> asyncio.Semaphore:
    # Created lazily so it binds to the running event loop.
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(config.MAX_CONCURRENT)
    return _semaphore


def ensure_clean_cwd() -> None:
    """Create the throwaway cwd used for every invocation (no CLAUDE.md leaks in)."""
    config.CLEAN_CWD.mkdir(parents=True, exist_ok=True)


def _mcp_config_json(token: str) -> str:
    """Inline --mcp-config payload attaching the configured MCP server for one user.
    Passed as a CLI string argument (the CLI accepts JSON files or strings)."""
    return json.dumps({
        "mcpServers": {
            config.MCP_SERVER_NAME: {
                "type": "http",
                "url": config.MCP_SERVER_URL,
                "headers": {"Authorization": f"Bearer {token}"},
            }
        }
    })


# Every built-in `claude` tool, hard-denied when an MCP server is attached so the
# model can reach ONLY the MCP server's tools (not Bash/Read/Write on the gateway
# host). --disallowedTools overrides --permission-mode bypassPermissions. Keep this
# list in sync if the CLI gains new built-ins.
_BUILTIN_TOOLS = (
    "Task,Bash,BashOutput,KillShell,KillBash,Glob,Grep,Read,Edit,Write,"
    "MultiEdit,NotebookEdit,NotebookRead,WebFetch,WebSearch,TodoWrite,"
    "SlashCommand,ExitPlanMode,ListMcpResources,ReadMcpResource"
)


def build_argv(req: CanonicalRequest) -> list[str]:
    """Assemble the contamination-neutralized `claude` command line.

    Plain chat runs with `--tools ""` (no tools). When a per-user MCP token is
    present and MCP is enabled, we run `--tools default` (the only value that
    surfaces MCP tools) but hard-deny every built-in via `--disallowedTools`, so
    only the one configured company-data server's tools are reachable and no
    built-in tool (Bash/Read/Write) can touch the gateway host.

    The token rides in the inline --mcp-config JSON (visible in this process's argv
    on the gateway host — acceptable on the dedicated single-tenant gateway VM;
    switch to a 0600 temp-file config if that host ever becomes multi-tenant).
    """
    argv = [
        "claude", "-p",
        "--model", req.model,
        "--input-format", "stream-json",
        "--output-format", "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--no-session-persistence",
        "--setting-sources", "",  # do not load user/project/local settings (where hooks live)
        "--system-prompt", req.system or config.DEFAULT_SYSTEM_PROMPT,
    ]
    if config.mcp_enabled() and req.mcp_token:
        # `--tools default` is the ONLY value that surfaces MCP tools to the model:
        # "" (the previous value) strips MCP tools too, and specific/MCP names are
        # treated as built-in names (MCP excluded). "default" also enables built-ins,
        # so we hard-deny every built-in via --disallowedTools (which overrides
        # bypassPermissions), leaving ONLY this MCP server's tools reachable. This is
        # what actually keeps the gateway host isolated (Bash/Read/Write can't run).
        argv += [
            "--mcp-config", _mcp_config_json(req.mcp_token),
            "--strict-mcp-config",  # ignore all other MCP configs on the machine
            "--tools", "default",
            "--disallowedTools", _BUILTIN_TOOLS,  # host isolation (overrides bypass)
            "--allowedTools", f"mcp__{config.MCP_SERVER_NAME}",
            "--permission-mode", "bypassPermissions",  # safe: built-ins are hard-denied
        ]
    else:
        argv += ["--tools", ""]  # plain chat: no tools at all
    effort = models.resolve_effort(req.model)
    if effort:
        argv += ["--effort", effort]
    if config.ISOLATION_MODE == "bare":
        argv.append("--bare")
    return argv


def _media_to_cli(block: dict) -> dict | None:
    """Map an image/document content block to its CLI base64 source form (None otherwise)."""
    if block.get("type") in ("image", "document"):
        return {
            "type": block["type"],
            "source": {"type": "base64", "media_type": block["media_type"], "data": block["data"]},
        }
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


def _media_stats(req: CanonicalRequest) -> tuple[int, int, int]:
    """Count native image/document blocks and approx decoded bytes across all turns.

    Only blocks still present as native media reach the engine — the Anthropic
    surface flattens PDFs to text upstream (pdf_to_text_block), so ``docs`` here
    reflects native-vision PDFs (the expensive path), not text-extracted ones.
    """
    imgs = docs = nbytes = 0
    for m in (req.messages or []):
        for b in m.blocks:
            t = b.get("type")
            if t == "image":
                imgs += 1
            elif t == "document":
                docs += 1
            else:
                continue
            nbytes += (len(b.get("data") or "") * 3) // 4  # base64 → approx raw bytes
    return imgs, docs, nbytes


async def run_claude(req: CanonicalRequest) -> AsyncIterator[CanonicalEvent]:
    """Spawn `claude` for one stateless invocation and yield CanonicalEvents."""
    if config.ISOLATION_MODE == "bare" and not config.ANTHROPIC_API_KEY:
        yield Error(500, "ISOLATION_MODE=bare requires ANTHROPIC_API_KEY in the environment")
        return

    ensure_clean_cwd()
    argv = build_argv(req)
    stdin_data = build_stdin(req)
    img_n, doc_n, media_n = _media_stats(req)  # usage accounting (see _log_outcome)

    # Fast-tier-only thinking control. req.model is the RESOLVED --model (the adapters
    # set model=resolve_model(...)), matching resolve_effort's contract in build_argv.
    # Only when a value is configured (per-model map or global) do we build an env
    # dict overriding the CLI's MAX_THINKING_TOKENS; otherwise env stays None so the
    # subprocess inherits the gateway env unchanged — opus/sonnet keep their thinking.
    mtt = models.resolve_max_thinking_tokens(req.model)
    subprocess_env = None
    if mtt is not None:
        subprocess_env = {**os.environ, "MAX_THINKING_TOKENS": str(mtt)}
        logger.info("thinking_disabled model=%s max_thinking_tokens=%s surface=%s",
                    req.model, mtt, req.surface or "-")

    # StreamReader line limit for the CLI's stdout/stderr. With --verbose the CLI
    # echoes the user message (inline base64 media included) as one NDJSON line, so
    # the limit must clear the payload we just sent. Scale to the actual stdin size
    # (2x for re-serialization slack) with a generous floor for other chatter — a
    # fixed cap would break multi-image requests (issue #11).
    stream_limit = max(config.STREAM_LIMIT, 2 * len(stdin_data))

    async with _get_semaphore():
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(config.CLEAN_CWD),
                limit=stream_limit,
                env=subprocess_env,  # None -> inherit gateway env (opus/sonnet untouched)
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
        cap_cache_read = None
        cap_cache_creation = None

        try:
            while True:
                remaining = config.TIMEOUT - (loop.time() - start)
                if remaining <= 0:
                    process.kill()
                    await process.wait()
                    _log_outcome("timeout", req, loop.time() - start,
                                 num_images=img_n, num_docs=doc_n, media_bytes=media_n,
                                 level=logging.WARNING)
                    yield Error(504, "upstream timeout")
                    return
                try:
                    line = await asyncio.wait_for(process.stdout.readline(), timeout=remaining)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    _log_outcome("timeout", req, loop.time() - start,
                                 num_images=img_n, num_docs=doc_n, media_bytes=media_n,
                                 level=logging.WARNING)
                    yield Error(504, "upstream timeout")
                    return
                except (ValueError, asyncio.LimitOverrunError) as e:
                    # readline() re-raises LimitOverrunError as ValueError when one
                    # stream-json line exceeds the reader limit (issue #11). The line
                    # length isn't recoverable from the exception, so log the limit.
                    process.kill()
                    await process.wait()
                    tail = b""
                    try:
                        tail = await process.stderr.read()
                    except Exception:
                        pass
                    logger.error(
                        "stream line exceeded limit=%d after %.1fs model=%s: %s; stderr tail: %s",
                        stream_limit, loop.time() - start, req.model, e,
                        tail.decode("utf-8", errors="replace").strip()[-500:],
                    )
                    yield Error(502, f"upstream stream line exceeded gateway limit "
                                     f"({stream_limit} bytes); raise STREAM_LIMIT if legitimate")
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
                        cap_cache_read = usage.get("cache_read_input_tokens", cap_cache_read)
                        cap_cache_creation = usage.get("cache_creation_input_tokens", cap_cache_creation)
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
                        if u.get("cache_read_input_tokens") is not None:
                            cap_cache_read = u["cache_read_input_tokens"]
                        if u.get("cache_creation_input_tokens") is not None:
                            cap_cache_creation = u["cache_creation_input_tokens"]
                elif otype == "result":
                    if obj.get("is_error") or obj.get("subtype") != "success":
                        await process.wait()
                        _log_outcome("error", req, loop.time() - start,
                                     num_images=img_n, num_docs=doc_n, media_bytes=media_n,
                                     level=logging.WARNING)
                        yield Error(502, obj.get("result") or "upstream error")
                        return
                    usage = obj.get("usage", {})
                    in_tok = usage.get("input_tokens", cap_in or 0)
                    out_tok = usage.get("output_tokens", cap_out or 0)
                    cache_read = usage.get("cache_read_input_tokens", cap_cache_read)
                    cache_creation = usage.get("cache_creation_input_tokens", cap_cache_creation)
                    await process.wait()
                    _log_outcome("success", req, loop.time() - start, in_tok, out_tok,
                                 cache_read=cache_read, cache_creation=cache_creation,
                                 num_images=img_n, num_docs=doc_n, media_bytes=media_n)
                    yield Stop(
                        stop_reason=map_stop_reason(obj.get("stop_reason") or cap_stop),
                        output_tokens=out_tok,
                        input_tokens=in_tok,
                    )
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
                _log_outcome("no-output", req, loop.time() - start,
                             num_images=img_n, num_docs=doc_n, media_bytes=media_n,
                             level=logging.WARNING)
                yield Error(502, msg or "no output from claude")
            else:
                _log_outcome("success", req, loop.time() - start, cap_in or 0, cap_out or 0,
                             cache_read=cap_cache_read, cache_creation=cap_cache_creation,
                             num_images=img_n, num_docs=doc_n, media_bytes=media_n)
                yield Stop(stop_reason=map_stop_reason(cap_stop),
                           output_tokens=cap_out or 0, input_tokens=cap_in or 0)
        except Exception as e:  # noqa: BLE001 - surface any spawn/read failure as an error event
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass
            _log_outcome("exception", req, loop.time() - start,
                         num_images=img_n, num_docs=doc_n, media_bytes=media_n,
                         level=logging.ERROR)
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
