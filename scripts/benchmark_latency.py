#!/usr/bin/env python3
"""Measure gateway end-to-end latency without printing prompts or credentials.

Example:
  python scripts/benchmark_latency.py https://host/v1/messages \
    --model claude-haiku-4-5-20251001 --requests 20 --concurrency 2

API_KEY is read from the environment by default. Set MCP_TOKEN only when the
benchmark should intentionally measure the MCP path.
"""
import argparse
import asyncio
import json
import os
import time

import httpx
from dotenv import load_dotenv


def percentile(values, pct):
    values = sorted(float(v) for v in values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def summarize(samples):
    ok = [sample for sample in samples if sample["status"] == 200 and sample["ttft_ms"] is not None]
    def metric(field, pct):
        value = percentile([sample[field] for sample in ok], pct)
        return round(value) if value is not None else None
    return {
        "requests": len(samples),
        "successful": len(ok),
        "errors": len(samples) - len(ok),
        "status_counts": {str(code): sum(1 for s in samples if s["status"] == code)
                          for code in sorted({s["status"] for s in samples})},
        "ttft_ms": {"p50": metric("ttft_ms", 0.50), "p95": metric("ttft_ms", 0.95),
                    "p99": metric("ttft_ms", 0.99)},
        "total_ms": {"p50": metric("total_ms", 0.50), "p95": metric("total_ms", 0.95),
                     "p99": metric("total_ms", 0.99)},
    }


async def run(args):
    api_key = os.getenv(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"missing credential in ${args.api_key_env}")
    mcp_token = os.getenv(args.mcp_token_env, "") if args.mcp else ""
    if args.mcp and not mcp_token:
        raise SystemExit(f"--mcp requires ${args.mcp_token_env}")

    headers = {
        "x-api-key": api_key,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    if mcp_token:
        headers["x-mcp-token"] = mcp_token
    payload = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "stream": True,
        "messages": [{"role": "user", "content": args.prompt}],
    }
    sem = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)

    async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:
        async def one():
            async with sem:
                started = time.perf_counter()
                first_text = None
                status = 0
                try:
                    async with client.stream("POST", args.url, headers=headers, json=payload) as response:
                        status = response.status_code
                        async for line in response.aiter_lines():
                            if first_text is None and "content_block_delta" in line and "text_delta" in line:
                                first_text = time.perf_counter()
                    ended = time.perf_counter()
                    return {"status": status,
                            "ttft_ms": round((first_text - started) * 1000) if first_text else None,
                            "total_ms": round((ended - started) * 1000)}
                except (httpx.HTTPError, asyncio.TimeoutError):
                    return {"status": status, "ttft_ms": None,
                            "total_ms": round((time.perf_counter() - started) * 1000)}

        for _ in range(args.warmup):
            await one()
        samples = await asyncio.gather(*(one() for _ in range(args.requests)))
    result = {"url_origin": httpx.URL(args.url).copy_with(path="/").human_repr(),
              "model": args.model, "mcp": bool(args.mcp),
              "concurrency": args.concurrency, **summarize(samples)}
    print(json.dumps(result, indent=2))
    return 0 if result["errors"] == 0 else 2


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="full Anthropic Messages endpoint")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prompt", default="Reply with OK and nothing else.")
    parser.add_argument("--api-key-env", default="API_KEY")
    parser.add_argument("--mcp", action="store_true")
    parser.add_argument("--mcp-token-env", default="MCP_TOKEN")
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1 or args.warmup < 0:
        parser.error("requests/concurrency must be positive and warmup non-negative")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
