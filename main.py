"""Run sequential requests against an independently started frontend."""

import csv
import argparse
import gzip
import http.client
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue

from rich.live import Live
from rich.progress import Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from config import OUTPUT_CSV
from dashboard import (
    EMERALD,
    WHITE,
    CYAN,
    ROYAL_BLUE,
    ThickBarColumn,
    build_dashboard,
    console,
    benchmark_title,
)
from logic import send_request
from prompts import build_requests


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the request latency benchmark.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--cache-reuse",
        action="store_true",
        help="Run the fixed six-request A/B prefix sequence",
    )
    selection.add_argument(
        "--round-robin",
        action="store_true",
        help="Run only the round-robin setup",
    )
    selection.add_argument(
        "--kv-router",
        action="store_true",
        help="Run only the KV-router setup",
    )

    args = parser.parse_args(argv)

    if args.cache_reuse:
        return ["cache-reuse"]
    if args.round_robin:
        return ["round-robin"]
    if args.kv_router:
        return ["kv-router"]
    return None


def choose_experiments():
    """Show an interactive menu when no CLI selector is provided."""
    options = {
        "1": "cache-reuse",
        "2": "round-robin",
        "3": "kv-router",
    }

    console.print("\n[bold]Select the active experiment:[/]")
    console.print("  1) Cache-Reuse Test")
    console.print("  2) Round Robin")
    console.print("  3) KV Router")
    console.print("Start the matching frontend separately before continuing.")

    try:
        choice = input("Selection [1-3, q to quit]: ").strip().lower()
    except EOFError:
        console.print(
            "\n[dim]No interactive input available. "
            "Use --cache-reuse, --round-robin, or --kv-router.[/]"
        )
        return []

    if choice in {"q", "quit"}:
        return []

    if choice not in options:
        console.print("[red]Choose 1, 2, 3, or q.[/]")
        return choose_experiments()

    return [options[choice]]


def _latest_request_end(
    trace_prefix, request_started_at_ms=None, request_completed_at_ms=None
):
    """Return the newest matching request_end fingerprint and worker IDs."""
    prefix_path = Path(trace_prefix)
    pattern = re.compile(rf"{re.escape(prefix_path.name)}\.(\d+)\.jsonl\.gz")
    matches = (
        (int(match.group(1)), path)
        for path in prefix_path.parent.glob(f"{prefix_path.name}.*.jsonl.gz")
        if (match := pattern.fullmatch(path.name))
    )
    latest = max(matches, default=None, key=lambda item: item[0])
    if latest is None:
        return None

    try:
        with gzip.open(latest[1], "rt", encoding="utf-8") as trace_file:
            lines = trace_file.readlines()
    except (OSError, EOFError, UnicodeError):
        return None

    for line in reversed(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        event = record.get("event") if isinstance(record, dict) else None
        if not isinstance(event, dict) or event.get("event_type") != "request_end":
            continue
        request = event.get("request")
        if request_started_at_ms is not None:
            request_received_ms = (
                request.get("request_received_ms")
                if isinstance(request, dict)
                else None
            )
            if (
                not isinstance(request_received_ms, (int, float))
                or request_received_ms < request_started_at_ms
                or (
                    request_completed_at_ms is not None
                    and request_received_ms > request_completed_at_ms
                )
            ):
                continue
        worker = request.get("worker") if isinstance(request, dict) else None
        if isinstance(worker, dict):
            prefill_worker_id = worker.get("prefill_worker_id", "")
            decode_worker_id = worker.get("decode_worker_id", "")
        else:
            prefill_worker_id = decode_worker_id = ""
        fingerprint = json.dumps(record, sort_keys=True, separators=(",", ":"))
        return (
            fingerprint,
            prefill_worker_id,
            decode_worker_id,
        )
    return None


def latest_request_metrics(
    trace_prefix,
    previous_event=None,
    request_started_at_ms=None,
    request_completed_at_ms=None,
    timeout=2.0,
):
    """Wait for the matching request_end event and return its worker IDs."""
    deadline = time.monotonic() + timeout
    while True:
        latest = _latest_request_end(
            trace_prefix,
            request_started_at_ms=request_started_at_ms,
            request_completed_at_ms=request_completed_at_ms,
        )
        if latest is not None and latest[0] != previous_event:
            return latest[1:]
        if time.monotonic() >= deadline:
            return "", ""
        time.sleep(0.05)


def _trace_prefix_for_experiment(experiment_name):
    return {
        "round-robin": "./traces/rr",
        "kv-router": "./traces/kv",
    }.get(experiment_name)


def _debug_log(message):
    with open("debug.txt", "a", encoding="utf-8") as debug_file:
        debug_file.write(f"{message}\n")


def _prefix_cache_metrics(port):
    metric_names = (
        "vllm:prefix_cache_hits_total",
        "vllm:prefix_cache_queries_total",
    )
    metrics = {}
    try:
        with urllib.request.urlopen(
            f"http://localhost:{port}/metrics", timeout=2
        ) as response:
            for line in response:
                fields = line.decode("utf-8").split()
                if len(fields) < 2:
                    continue
                for metric_name in metric_names:
                    if fields[0] == metric_name or fields[0].startswith(
                        f"{metric_name}{{"
                    ):
                        metrics[metric_name] = float(fields[1])
                        break
    except (
        urllib.error.URLError,
        http.client.HTTPException,
        TimeoutError,
        OSError,
        UnicodeError,
        ValueError,
    ) as exc:
        _debug_log(f"Could not scrape prefix-cache metrics on port {port}: {exc}")
        return None
    missing = set(metric_names) - metrics.keys()
    if missing:
        _debug_log(
            f"Prefix-cache metrics missing on port {port}: "
            f"{', '.join(sorted(missing))}"
        )
        return None
    return metrics


def _prefix_hit_rate(hits, queries):
    return hits / queries * 100 if queries else None


def _log_prefix_hit_rate_diagnostic(
    request_id,
    worker_id,
    previous_hits,
    previous_queries,
    current_hits,
    current_queries,
    delta_hits,
    delta_queries,
    hit_rate,
):
    _debug_log(
        f"request={request_id}, actual_prefill_worker={worker_id or 'N/A'}, "
        f"previous_hits={previous_hits if previous_hits is not None else 'N/A'}, "
        f"previous_queries={previous_queries if previous_queries is not None else 'N/A'}, "
        f"current_hits={current_hits if current_hits is not None else 'N/A'}, "
        f"current_queries={current_queries if current_queries is not None else 'N/A'}, "
        f"delta_hits={delta_hits if delta_hits is not None else 'N/A'}, "
        f"delta_queries={delta_queries if delta_queries is not None else 'N/A'}, "
        f"calculated_rate={f'{hit_rate:.2f}%' if hit_rate is not None else 'N/A'}"
    )


def _update_worker_metrics(worker_metrics, worker_id, metrics, request_id=None):
    worker = worker_metrics[worker_id]
    hits = metrics["vllm:prefix_cache_hits_total"]
    queries = metrics["vllm:prefix_cache_queries_total"]
    previous_hits = worker["hits"]
    previous_queries = worker["queries"]
    hit_rate = None
    delta_hits = None
    delta_queries = None
    if previous_hits is not None and previous_queries is not None:
        delta_hits = hits - previous_hits
        delta_queries = queries - previous_queries
        if delta_hits >= 0 and delta_queries >= 0:
            hit_rate = _prefix_hit_rate(delta_hits, delta_queries)
    if request_id is not None:
        _log_prefix_hit_rate_diagnostic(
            request_id,
            worker_id,
            previous_hits,
            previous_queries,
            hits,
            queries,
            delta_hits,
            delta_queries,
            hit_rate,
        )
    worker.update(
        hits=hits,
        queries=queries,
        prefix_hit_rate=f"{hit_rate:.2f}" if hit_rate is not None else "",
    )
    return worker


def _label_worker_id(worker_id, worker_labels):
    if worker_id in (None, ""):
        return ""
    worker_key = str(worker_id)
    if worker_key not in worker_labels:
        worker_labels[worker_key] = f"w{len(worker_labels)}"
    return worker_labels[worker_key]


def _record_prompt_workers(
    prompt, prefill_worker_id, decode_worker_id, worker_labels, prompt_workers
):
    cache_on_worker = prompt_workers.get(prompt, "-")
    prefill_worker = _label_worker_id(prefill_worker_id, worker_labels)
    decode_worker = _label_worker_id(decode_worker_id, worker_labels)
    if prefill_worker:
        prompt_workers[prompt] = prefill_worker
    return prefill_worker, decode_worker, cache_on_worker


def _active_request_seconds(results):
    intervals = sorted(
        (row["_request_started_at"], row["_response_received_at"])
        for row in results
        if row.get("_request_started_at") is not None
        and row.get("_response_received_at") is not None
    )
    if not intervals:
        return 0.0

    active_seconds = 0.0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start > end:
            active_seconds += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return active_seconds + end - start


def run_benchmark(experiment_name):
    results = []
    worker_labels = {}
    prompt_workers = {}
    worker_ports = {
        "w0": 8100,
        "w1": 8110,
    }
    worker_metrics = {
        "w0": {"hits": None, "queries": None, "prefix_hit_rate": ""},
        "w1": {"hits": None, "queries": None, "prefix_hit_rate": ""},
    }
    output_csv = OUTPUT_CSV.replace("roundrobin", experiment_name.replace("-", ""))
    console.print(
        f"\n[bold {ROYAL_BLUE}]Starting {benchmark_title(experiment_name)}...[/]"
    )
    progress = Progress(
        TextColumn(f"[bold {WHITE}]{{task.description}}[/]"),
        ThickBarColumn(width=24),
        TextColumn(f"[bold {CYAN}]{{task.percentage:>5.1f}}%[/]"),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=False,
    )
    workload = build_requests()
    task = progress.add_task("Requests processed", total=len(workload))
    events = Queue()

    def execute_request(request_id, question, prompt, family_id, request_role):
        events.put((request_id, prompt, "RUNNING", "", None, None))

        def on_token(text):
            events.put((request_id, prompt, "RUNNING", text, None, None))

        row = {
            "request_id": request_id,
            "family_id": family_id,
            "request_role": request_role,
            "status": "success",
            "ttft_ms": "",
            "response_ms": "",
            "total_ms": "",
            "cached_tokens": "",
            "prefix_cache_hits": "",
            "prefix_cache_queries": "",
            "prefix_hit_rate": "",
            "actual_prefill_worker": "",
            "actual_decode_worker": "",
            "cache_on_worker": "-",
            "_request_started_at": None,
            "_response_received_at": None,
            "error": "",
        }
        row["cache_on_worker"] = prompt_workers.get(prompt, "-")
        trace_prefix = _trace_prefix_for_experiment(experiment_name)
        try:
            request_started_at_ms = time.time() * 1000
            measurement = send_request(
                prompt, on_token=on_token, request_id=request_id
            )
            row["_request_started_at"] = measurement["request_started_at"]
            row["_response_received_at"] = measurement["response_received_at"]
            row["cached_tokens"] = measurement["cached_tokens"]
            for metric in ("ttft_ms", "response_ms", "total_ms"):
                row[metric] = measurement[metric]
            response_text = measurement["output"]
            phase = "COMPLETE"
        except Exception as exc:  # noqa: BLE001
            if row["_request_started_at"] is not None:
                row["_response_received_at"] = time.perf_counter()
            row.update(status="failed", error=str(exc))
            response_text = f"Request failed: {exc}"
            phase = "FAILED"
        request_completed_at_ms = time.time() * 1000
        if trace_prefix and row["status"] == "success":
            (
                raw_prefill_worker_id,
                raw_decode_worker_id,
            ) = latest_request_metrics(
                trace_prefix,
                request_started_at_ms=request_started_at_ms,
                request_completed_at_ms=request_completed_at_ms,
            )
            (
                row["actual_prefill_worker"],
                row["actual_decode_worker"],
                row["cache_on_worker"],
            ) = _record_prompt_workers(
                prompt,
                raw_prefill_worker_id,
                raw_decode_worker_id,
                worker_labels,
                prompt_workers,
            )
            prefill_worker = row["actual_prefill_worker"]
            worker_id = prefill_worker
            port = worker_ports.get(worker_id)
            if port is not None:
                metrics = _prefix_cache_metrics(port)
                if metrics is not None:
                    worker_snapshot = _update_worker_metrics(
                        worker_metrics, prefill_worker, metrics, request_id
                    )
                    if worker_snapshot["prefix_hit_rate"] != "":
                        _debug_log(
                            f"Prefix Hit Rate: "
                            f"{worker_snapshot['prefix_hit_rate']}%"
                        )
                    else:
                        _debug_log(
                            "Prefix Hit Rate: N/A (baseline, counter reset, "
                            "or zero queries)"
                        )
                    row["prefix_hit_rate"] = worker_snapshot["prefix_hit_rate"]
                else:
                    worker_snapshot = worker_metrics[prefill_worker]
                    _log_prefix_hit_rate_diagnostic(
                        request_id,
                        prefill_worker,
                        worker_snapshot["hits"],
                        worker_snapshot["queries"],
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                worker_snapshot = worker_metrics[prefill_worker]
                row["prefix_cache_hits"] = (
                    worker_snapshot["hits"]
                    if worker_snapshot["hits"] is not None
                    else ""
                )
                row["prefix_cache_queries"] = (
                    worker_snapshot["queries"]
                    if worker_snapshot["queries"] is not None
                    else ""
                )
            else:
                worker_snapshot = worker_metrics.get(prefill_worker, {})
                _log_prefix_hit_rate_diagnostic(
                    request_id,
                    prefill_worker,
                    worker_snapshot.get("hits"),
                    worker_snapshot.get("queries"),
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                if prefill_worker:
                    console.print(
                        f"[yellow]No metrics port configured for "
                        f"prefill worker {prefill_worker}.[/]"
                    )
        events.put(
            (
                request_id,
                prompt,
                phase,
                response_text,
                row,
                row["_response_received_at"],
            )
        )

    with (
        Live(
            build_dashboard(results, progress, "—", "READY", "", "", experiment_name),
            console=console,
            refresh_per_second=8,
            transient=False,
            vertical_overflow="visible",
        ) as live,
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        for request in workload:
            future = executor.submit(execute_request, *request)
            while not future.done() or not events.empty():
                try:
                    request_id, question, phase, text, row, _ = events.get(timeout=0.1)
                except Empty:
                    continue
                if row is not None:
                    results.append(row)
                    progress.advance(task)
                live.update(
                    build_dashboard(
                        results,
                        progress,
                        request_id,
                        phase,
                        question,
                        text,
                        experiment_name,
                    ),
                    refresh=False,
                )
            future.result()

    if results:
        with open(output_csv, "w", newline="", encoding="utf-8") as file:
            fieldnames = [key for key in results[0] if not key.startswith("_")]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows({key: row[key] for key in fieldnames} for row in results)
        console.print(
            f"\n[bold {EMERALD}]Saved {len(results)} request records to {output_csv}[/]"
        )
    batch_seconds = _active_request_seconds(results)
    console.print(
        f"Time with requests in flight: {batch_seconds:.2f} s "
        f"({len(workload)} requests, concurrency 1)"
    )


def main():
    selections = parse_args()
    if selections is None:
        selections = choose_experiments()
    if not selections:
        console.print("[dim]No benchmark selected.[/]")
        return
    for experiment_name in selections:
        run_benchmark(experiment_name)


if __name__ == "__main__":
    main()
