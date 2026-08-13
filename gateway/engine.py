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

# Two keyed concurrency lanes so long heavy extraction jobs (opus/sonnet, up to the
# 300s budget) can't starve latency-sensitive fast-tier (haiku) calls. Created
# lazily so each binds to the running event loop. Heavy keeps the historical
# MAX_CONCURRENT capacity unchanged; fast gets its own MAX_CONCURRENT_FAST.
_semaphores: dict[str, asyncio.Semaphore] = {}


_REASON_MAX = 200


def _short_reason(reason: str | None) -> str | None:
    """One-line, length-capped reason so a log line stays greppable."""
    if not reason:
        return None
    return " ".join(str(reason).split())[:_REASON_MAX] or None


def is_overloaded(message: str | None) -> bool:
    """True when the upstream reported saturation rather than a request problem.

    The CLI surfaces it as text ("API Error: 529 Overloaded…"), so text is all we
    have to match on. Callers should back off rather than retry immediately: on
    2026-07-29 a saturated upstream cost ~200s per attempt because the CLI retries
    internally before giving up.
    """
    if not message:
        return False
    return "529" in message or "overloaded" in message.lower()


def _log_outcome(outcome: str, req: CanonicalRequest, elapsed: float,
                 in_tok: int | None = None, out_tok: int | None = None,
                 cache_read: int | None = None, cache_creation: int | None = None,
                 num_images: int = 0, num_docs: int = 0, media_bytes: int = 0,
                 lane: str | None = None, queue_wait_ms: int | None = None,
                 spawn_ms: int | None = None,
                 stdin_ms: int | None = None, first_event_ms: int | None = None,
                 first_text_ms: int | None = None, total_ms: int | None = None,
                 prompt_bytes: int = 0, history_messages: int = 0,
                 mcp: bool = False,
                 reason: str | None = None,
                 level: int = logging.INFO) -> None:
    """One line per invocation so errors and durations are visible in journald,
    plus a structured JSONL record (when USAGE_LOG is set) for aggregation.

    ``reason`` is the upstream's own explanation of a non-success outcome. Log it:
    without it an operator sees only the word "error" and has to reproduce the
    request to find out what happened. That cost real time in the 2026-07-29
    ConstraAP incident — five ~200s `error` lines with no hint that the cause was
    `API Error: 529 Overloaded`, which was sitting in the 502 body all along.
    """
    logger.log(level,
               "run_claude %s surface=%s model=%s lane=%s mcp=%s queue_ms=%s spawn_ms=%s "
               "stdin_ms=%s first_event_ms=%s first_text_ms=%s total_ms=%s "
               "elapsed=%.1fs in=%s out=%s cache_read=%s cache_write=%s imgs=%s docs=%s%s",
               outcome, req.surface or "-", req.model, lane or "-", mcp, queue_wait_ms,
               spawn_ms, stdin_ms, first_event_ms, first_text_ms, total_ms,
               elapsed, in_tok, out_tok,
               cache_read, cache_creation, num_images, num_docs,
               f" reason={_short_reason(reason)}" if reason else "")
    usage_log.record(outcome=outcome, req=req, elapsed=elapsed,
                     input_tokens=in_tok, output_tokens=out_tok,
                     cache_read=cache_read, cache_creation=cache_creation,
                     num_images=num_images, num_docs=num_docs, media_bytes=media_bytes,
                     lane=lane, queue_wait_ms=queue_wait_ms, spawn_ms=spawn_ms,
                     stdin_ms=stdin_ms, first_event_ms=first_event_ms,
                     first_text_ms=first_text_ms, total_ms=total_ms,
                     prompt_bytes=prompt_bytes, history_messages=history_messages,
                     mcp=mcp,
                     reason=_short_reason(reason))


def _get_semaphore(lane: str) -> asyncio.Semaphore:
    # Created lazily so each binds to the running event loop. "fast" and "heavy"
    # are the only lanes; capacities come from config.
    sem = _semaphores.get(lane)
    if sem is None:
        cap = config.MAX_CONCURRENT_FAST if lane == "fast" else config.MAX_CONCURRENT
        sem = asyncio.Semaphore(cap)
        _semaphores[lane] = sem
    return sem


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
    effort = req.effort_override or models.resolve_effort(req.model)
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

    # Lane selection is on the RESOLVED --model (req.model), same contract as
    # resolve_effort / resolve_max_thinking_tokens: fast tier (haiku) gets its own
    # semaphore so it can't queue behind heavy extraction jobs on the heavy lane.
    lane = "fast" if models.is_fast_model(req.model) else "heavy"
    sem = _get_semaphore(lane)

    _loop = asyncio.get_event_loop()
    enqueue_t = _loop.time()
    # Bound the queue wait: don't let a saturated gateway silently sit on a request
    # for its whole timeout budget. On timeout, fail fast with a 503 the client can
    # react to (retry/backoff) immediately. acquired guards the release in finally.
    acquired = False
    try:
        await asyncio.wait_for(sem.acquire(), timeout=config.QUEUE_WAIT_MAX)
        acquired = True
    except asyncio.TimeoutError:
        queue_wait_ms = int((_loop.time() - enqueue_t) * 1000)
        _log_outcome("saturated", req, _loop.time() - enqueue_t,
                     num_images=img_n, num_docs=doc_n, media_bytes=media_n,
                     lane=lane, queue_wait_ms=queue_wait_ms,
                     total_ms=queue_wait_ms, prompt_bytes=len(stdin_data),
                     history_messages=max(0, len(req.messages or []) - 1),
                     mcp=bool(config.mcp_enabled() and req.mcp_token),
                     level=logging.WARNING)
        yield Error(503, "gateway saturated, retry")
        return

    queue_wait_ms = int((_loop.time() - enqueue_t) * 1000)
    try:
        spawn_t = _loop.time()
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
        spawn_ms = int((_loop.time() - spawn_t) * 1000)

        stdin_ms = 0
        first_event_ms = None
        first_text_ms = None

        # Close over phase timings so every outcome is directly usable for
        # percentile analysis without logging content or credentials.
        def _log(outcome, elapsed, *a, **kw):
            kw.setdefault("lane", lane)
            kw.setdefault("queue_wait_ms", queue_wait_ms)
            kw.setdefault("spawn_ms", spawn_ms)
            kw.setdefault("stdin_ms", stdin_ms)
            kw.setdefault("first_event_ms", first_event_ms)
            kw.setdefault("first_text_ms", first_text_ms)
            kw.setdefault("total_ms", int((_loop.time() - enqueue_t) * 1000))
            kw.setdefault("prompt_bytes", len(stdin_data))
            kw.setdefault("history_messages", max(0, len(req.messages or []) - 1))
            kw.setdefault("mcp", bool(config.mcp_enabled() and req.mcp_token))
            _log_outcome(outcome, req, elapsed, *a, **kw)

        stdin_t = _loop.time()
        try:
            process.stdin.write(stdin_data)
            await process.stdin.drain()
            process.stdin.close()
        except (BrokenPipeError, ConnectionResetError):
            pass
        stdin_ms = int((_loop.time() - stdin_t) * 1000)

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
                    _log("timeout", loop.time() - start,
                                 num_images=img_n, num_docs=doc_n, media_bytes=media_n,
                                 level=logging.WARNING)
                    yield Error(504, "upstream timeout")
                    return
                try:
                    line = await asyncio.wait_for(process.stdout.readline(), timeout=remaining)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    _log("timeout", loop.time() - start,
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
                        if first_event_ms is None:
                            first_event_ms = int((loop.time() - enqueue_t) * 1000)
                        msg = ev.get("message", {})
                        usage = msg.get("usage", {})
                        cap_in = usage.get("input_tokens")
                        cap_cache_read = usage.get("cache_read_input_tokens", cap_cache_read)
                        cap_cache_creation = usage.get("cache_creation_input_tokens", cap_cache_creation)
                        # MCP tool use can produce multiple internal Claude turns,
                        # each with its own message_start/message_stop. This HTTP
                        # request is one public Messages API response, so expose a
                        # single Start and append later text deltas to it until the
                        # final CLI result supplies the one public Stop.
                        if not started:
                            started = True
                            yield Start(model=msg.get("model"), input_tokens=usage.get("input_tokens", 0))
                    elif etype == "content_block_delta":
                        delta = ev.get("delta", {})
                        if delta.get("type") == "text_delta":
                            if first_text_ms is None:
                                first_text_ms = int((loop.time() - enqueue_t) * 1000)
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
                        msg = obj.get("result") or "upstream error"
                        # An overloaded upstream is NOT a generic bad gateway: the
                        # caller should back off and retry later, and conflating it
                        # with 502 hid that for a long time. Surface 529 so clients
                        # can branch on the status instead of grepping the message.
                        overloaded = is_overloaded(msg)
                        _log("overloaded" if overloaded else "error",
                                     loop.time() - start,
                                     num_images=img_n, num_docs=doc_n, media_bytes=media_n,
                                     reason=msg,
                                     level=logging.WARNING)
                        yield Error(529 if overloaded else 502, msg)
                        return
                    usage = obj.get("usage", {})
                    in_tok = usage.get("input_tokens", cap_in or 0)
                    out_tok = usage.get("output_tokens", cap_out or 0)
                    cache_read = usage.get("cache_read_input_tokens", cap_cache_read)
                    cache_creation = usage.get("cache_creation_input_tokens", cap_cache_creation)
                    await process.wait()
                    _log("success", loop.time() - start, in_tok, out_tok,
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
                _log("no-output", loop.time() - start,
                             num_images=img_n, num_docs=doc_n, media_bytes=media_n,
                             reason=msg,
                             level=logging.WARNING)
                yield Error(502, msg or "no output from claude")
            else:
                _log("success", loop.time() - start, cap_in or 0, cap_out or 0,
                             cache_read=cap_cache_read, cache_creation=cap_cache_creation,
                             num_images=img_n, num_docs=doc_n, media_bytes=media_n)
                yield Stop(stop_reason=map_stop_reason(cap_stop),
                           output_tokens=cap_out or 0, input_tokens=cap_in or 0)
        except asyncio.CancelledError:
            # A3: the client disconnected / the driving task was cancelled. The plain
            # `except Exception` below does NOT catch CancelledError, so without this
            # the CLI subprocess would keep running to the gateway TIMEOUT while still
            # holding this lane's slot — self-amplifying congestion. Kill it now, log,
            # and re-raise so the slot frees immediately (release is in the finally).
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass
            _log("cancelled", loop.time() - start,
                 num_images=img_n, num_docs=doc_n, media_bytes=media_n,
                 level=logging.WARNING)
            raise
        except Exception as e:  # noqa: BLE001 - surface any spawn/read failure as an error event
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass
            _log("exception", loop.time() - start,
                         num_images=img_n, num_docs=doc_n, media_bytes=media_n,
                         reason=str(e),
                         level=logging.ERROR)
            yield Error(500, str(e))
    finally:
        # Guaranteed slot release. Only release what we acquired (the queue-wait
        # timeout path returns before entering this try, with acquired=False).
        if acquired:
            sem.release()


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
