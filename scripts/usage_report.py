#!/usr/bin/env python3
"""Aggregate the gateway usage log (JSONL written by gateway/usage_log.py).

Usage:
    python scripts/usage_report.py [PATH] [--today] [--since YYYY-MM-DD] [--top N]

PATH defaults to $USAGE_LOG, then ./usage.jsonl. One JSON object per line; lines
that don't parse are skipped. Costs are the REFERENCE figures from pricing.py
(the gateway runs on a subscription by default — see pricing.py).
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone


def _load(path, since):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if since and (r.get("ts") or "") < since:
                continue
            rows.append(r)
    return rows


def _fmt_int(n):
    return f"{int(n):,}"


def _percentile(values, pct):
    values = sorted(float(v) for v in values if v is not None)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def _latency_stats(rows):
    def pct(field, p):
        value = _percentile([r.get(field) for r in rows], p)
        return round(value) if value is not None else None
    return {
        "calls": len(rows),
        "queue_p95_ms": pct("queue_wait_ms", 0.95),
        "spawn_p95_ms": pct("spawn_ms", 0.95),
        "ttft_p50_ms": pct("first_text_ms", 0.50),
        "ttft_p95_ms": pct("first_text_ms", 0.95),
        "ttft_p99_ms": pct("first_text_ms", 0.99),
        "total_p50_ms": pct("total_ms", 0.50),
        "total_p95_ms": pct("total_ms", 0.95),
        "total_p99_ms": pct("total_ms", 0.99),
    }


def _print_latency(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[f"{row.get('model') or '-'}|{'mcp' if row.get('mcp') else 'plain'}"].append(row)
    print("\nLatency by model and MCP (milliseconds):")
    print(f"  {'key':<28} {'calls':>6} {'q95':>7} {'sp95':>7} "
          f"{'tt50':>7} {'tt95':>7} {'tt99':>7} {'tot50':>7} {'tot95':>7} {'tot99':>7}")
    for key, grouped in sorted(groups.items()):
        s = _latency_stats(grouped)
        val = lambda name: "-" if s[name] is None else str(s[name])
        print(f"  {key:<28} {s['calls']:>6} {val('queue_p95_ms'):>7} {val('spawn_p95_ms'):>7} "
              f"{val('ttft_p50_ms'):>7} {val('ttft_p95_ms'):>7} {val('ttft_p99_ms'):>7} "
              f"{val('total_p50_ms'):>7} {val('total_p95_ms'):>7} {val('total_p99_ms'):>7}")


def _bucket(rows, key):
    agg = defaultdict(lambda: {"n": 0, "in": 0, "out": 0, "cr": 0, "cw": 0,
                               "cost": 0.0, "docs": 0, "imgs": 0, "elapsed": 0.0})
    for r in rows:
        a = agg[r.get(key) or "-"]
        a["n"] += 1
        a["in"] += r.get("input_tokens", 0)
        a["out"] += r.get("output_tokens", 0)
        a["cr"] += r.get("cache_read", 0)
        a["cw"] += r.get("cache_creation", 0)
        a["cost"] += r.get("est_cost_usd") or 0.0
        a["docs"] += r.get("num_docs", 0)
        a["imgs"] += r.get("num_images", 0)
        a["elapsed"] += r.get("elapsed_s", 0.0)
    return agg


def _print_bucket(title, agg):
    print(f"\n{title}")
    print(f"  {'key':<16} {'calls':>7} {'in':>12} {'out':>10} "
          f"{'cache_rd':>10} {'cache_wr':>10} {'docs':>6} {'imgs':>6} {'~cost$':>10}")
    for k, a in sorted(agg.items(), key=lambda kv: kv[1]["cost"], reverse=True):
        print(f"  {k:<16} {a['n']:>7} {_fmt_int(a['in']):>12} {_fmt_int(a['out']):>10} "
              f"{_fmt_int(a['cr']):>10} {_fmt_int(a['cw']):>10} {a['docs']:>6} {a['imgs']:>6} "
              f"{a['cost']:>10.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=os.getenv("USAGE_LOG") or "usage.jsonl")
    ap.add_argument("--today", action="store_true", help="only today's UTC records")
    ap.add_argument("--since", help="ISO date/time lower bound, e.g. 2026-07-22")
    ap.add_argument("--top", type=int, default=10, help="show N most expensive calls")
    args = ap.parse_args()

    since = args.since
    if args.today:
        since = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if not os.path.exists(args.path):
        sys.exit(f"usage log not found: {args.path} (set USAGE_LOG or pass a path)")

    rows = _load(args.path, since)
    if not rows:
        print(f"No records in {args.path}" + (f" since {since}" if since else ""))
        return

    tot = {"in": sum(r.get("input_tokens", 0) for r in rows),
           "out": sum(r.get("output_tokens", 0) for r in rows),
           "cr": sum(r.get("cache_read", 0) for r in rows),
           "cw": sum(r.get("cache_creation", 0) for r in rows),
           "cost": sum((r.get("est_cost_usd") or 0.0) for r in rows),
           "docs": sum(r.get("num_docs", 0) for r in rows),
           "imgs": sum(r.get("num_images", 0) for r in rows)}
    prompt_total = tot["in"] + tot["cr"] + tot["cw"]
    hit = (tot["cr"] / prompt_total * 100) if prompt_total else 0.0

    print("=" * 70)
    print(f"Claude Gateway usage — {args.path}" + (f" (since {since})" if since else ""))
    print("=" * 70)
    print(f"  invocations       : {_fmt_int(len(rows))}")
    print(f"  input (uncached)  : {_fmt_int(tot['in'])} tok")
    print(f"  cache read        : {_fmt_int(tot['cr'])} tok")
    print(f"  cache write       : {_fmt_int(tot['cw'])} tok")
    print(f"  output            : {_fmt_int(tot['out'])} tok")
    print(f"  cache hit ratio   : {hit:.1f}%  (cache_read / total prompt tokens)")
    print(f"  native docs/imgs  : {tot['docs']} docs, {tot['imgs']} images")
    print(f"  reference cost    : ${tot['cost']:.3f}  (subscription = not billed; see pricing.py)")

    _print_bucket("By model:", _bucket(rows, "model"))
    _print_bucket("By surface:", _bucket(rows, "surface"))
    _print_bucket("By outcome:", _bucket(rows, "outcome"))
    _print_latency(rows)

    ranked = sorted(rows, key=lambda r: r.get("est_cost_usd") or 0.0, reverse=True)[:args.top]
    print(f"\nTop {len(ranked)} calls by reference cost:")
    for r in ranked:
        print(f"  {r.get('ts','?')}  {(r.get('surface') or '-'):<9} {(r.get('model') or '-'):<18} "
              f"in={_fmt_int(r.get('input_tokens',0))} out={_fmt_int(r.get('output_tokens',0))} "
              f"docs={r.get('num_docs',0)} imgs={r.get('num_images',0)} "
              f"~${r.get('est_cost_usd') or 0.0:.4f}")


if __name__ == "__main__":
    main()
