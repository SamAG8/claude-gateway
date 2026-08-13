"""Pure checks for latency percentile/report helpers."""
from scripts.benchmark_latency import percentile, summarize
from scripts.usage_report import _latency_stats


def test_percentile_interpolates_and_handles_single_value():
    assert percentile([10], 0.95) == 10
    assert percentile([10, 20, 30], 0.5) == 20
    assert percentile([10, 20, 30], 0.95) == 29


def test_benchmark_summary_excludes_errors_from_latency():
    summary = summarize([
        {"status": 200, "ttft_ms": 100, "total_ms": 150},
        {"status": 200, "ttft_ms": 200, "total_ms": 260},
        {"status": 503, "ttft_ms": None, "total_ms": 20},
    ])
    assert summary["successful"] == 2
    assert summary["errors"] == 1
    assert summary["ttft_ms"]["p50"] == 150
    assert summary["status_counts"] == {"200": 2, "503": 1}


def test_usage_latency_stats_exposes_gateway_phases():
    stats = _latency_stats([
        {"queue_wait_ms": 1, "spawn_ms": 4, "first_text_ms": 100, "total_ms": 160},
        {"queue_wait_ms": 3, "spawn_ms": 6, "first_text_ms": 200, "total_ms": 280},
    ])
    assert stats["calls"] == 2
    assert stats["queue_p95_ms"] == 3
    assert stats["spawn_p95_ms"] == 6
    assert stats["ttft_p50_ms"] == 150
    assert stats["total_p50_ms"] == 220
