"""Run concurrent benchmark request pairs against an independently started frontend."""

import csv
import argparse
import http.client
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from threading import Lock

from rich.live import Live
from rich.progress import Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from config import CONCURRENCY, OUTPUT_CSV, PROMPTING_TYPE, RUNS, WORKER_PORTS
from dashboard import (
    EMERALD,
    WHITE,
    CYAN,
    ROYAL_BLUE,
    INDIGO,
    ThickBarColumn,
    build_dashboard,
    console,
    benchmark_title,
)
from logic import send_request
from prompts import build_requests
from worker_discovery import latest_request_trace
from warmup import run_warmup


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the request latency benchmark.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--cache-reuse",
        action="store_true",
        help="Run the cache-reuse prefix sequence",
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


def _trace_prefix_for_experiment(experiment_name):
    return {
        "round-robin": "./traces/rr",
        "kv-router": "./traces/kv",
    }.get(experiment_name)


def _request_prefix_hit_rate(cached_tokens, prompt_tokens):
    if not isinstance(cached_tokens, (int, float)):
        return None
    if not isinstance(prompt_tokens, (int, float)) or prompt_tokens <= 0:
        return None
    return cached_tokens / prompt_tokens * 100


def _label_worker_id(worker_id, worker_labels):
    if worker_id in (None, ""):
        return ""
    worker_key = str(worker_id)
    if worker_key not in worker_labels:
        worker_labels[worker_key] = f"w{len(worker_labels)}"
    return worker_labels[worker_key]


def _format_previous_workers(workers):
    return "/".join(workers) if workers else "-"


def _record_prompt_workers(
    prompt, prefill_worker_id, decode_worker_id, worker_labels, prompt_workers
):
    previous_workers = prompt_workers.setdefault(prompt, [])
    previous_workers_label = _format_previous_workers(previous_workers)
    prefill_worker = _label_worker_id(prefill_worker_id, worker_labels)
    decode_worker = _label_worker_id(decode_worker_id, worker_labels)
    if prefill_worker and prefill_worker not in previous_workers:
        previous_workers.append(prefill_worker)
    return prefill_worker, decode_worker, previous_workers_label


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


def _reset_prefix_caches(worker_ports):
    for worker_id, port in worker_ports.items():
        request = urllib.request.Request(
            f"http://localhost:{port}/reset_prefix_cache",
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                body = response.read().decode("utf-8")
            try:
                result = json.loads(body)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Cache reset for {worker_id} on port {port} returned "
                    f"invalid JSON: {body[:1000]}"
                ) from exc
            if not isinstance(result, dict) or result.get("success") is not True:
                raise RuntimeError(
                    f"Cache reset failed for {worker_id} on port {port}: "
                    f"{body[:1000]}"
                )
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Could not reset prefix cache for {worker_id} on port {port}: "
                f"HTTP {exc.code}: {details[:1000]}"
            ) from exc
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            TimeoutError,
            OSError,
        ) as exc:
            raise RuntimeError(
                f"Could not reset prefix cache for {worker_id} on port {port}: {exc}"
            ) from exc
        console.print(f"Reset prefix cache on {worker_id} port {port}")


def run_benchmark(experiment_name, worker_labels):
    all_results = []
    output_name = OUTPUT_CSV.replace("roundrobin", experiment_name.replace("-", ""))
    output_path = Path(output_name)
    concurrency_prefix = "con_" if CONCURRENCY > 1 else ""
    output_csv = output_path.with_name(
        f"{concurrency_prefix}{PROMPTING_TYPE}_{output_path.name}"
    )
    for run_number in range(1, RUNS + 1):
        console.print(
            f"\n[bold {INDIGO}]Run {run_number}/{RUNS}: flushing worker caches[/]"
        )
        _reset_prefix_caches(WORKER_PORTS)
        run_results = _run_benchmark_once(
            experiment_name,
            worker_labels,
            run_number,
        )
        all_results.extend(run_results)

    if all_results:
        fieldnames = [key for key in all_results[0] if not key.startswith("_")]
        file_exists = output_csv.exists() and output_csv.stat().st_size > 0
        if file_exists:
            with open(output_csv, "r", newline="", encoding="utf-8") as existing_file:
                existing_fieldnames = next(csv.reader(existing_file), [])
            if existing_fieldnames != fieldnames:
                raise RuntimeError(
                    f"Cannot append results to {output_csv}: CSV columns do not "
                    "match the existing file"
                )
        with open(output_csv, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(
                {key: row[key] for key in fieldnames}
                for row in all_results
            )
        console.print(
            f"\n[bold {EMERALD}]Saved {len(all_results)} request records "
            f"across {RUNS} runs to {output_csv}[/]"
        )
    batch_seconds = _active_request_seconds(all_results)
    console.print(
        f"Time with requests in flight: {batch_seconds:.2f} s "
        f"({len(all_results)} requests, concurrency {CONCURRENCY})"
    )


def _run_benchmark_once(experiment_name, worker_labels, run_number):
    results = []
    prompt_workers = {}
    console.print(
        f"\n[bold {ROYAL_BLUE}]Starting {benchmark_title(experiment_name)} "
        f"run {run_number}/{RUNS}...[/]"
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
    workload = build_requests(run_number)
    task = progress.add_task(
        f"Run {run_number}/{RUNS} requests",
        total=len(workload),
    )
    events = Queue()
    metrics_lock = Lock()

    def execute_request(request_id, question, prompt, family_id, _request_role):
        events.put((request_id, prompt, "RUNNING", "", None, None))

        def on_token(text):
            events.put((request_id, prompt, "RUNNING", text, None, None))

        row = {
            "run_number": run_number,
            "request_id": request_id,
            "family_id": family_id,
            "Mode": PROMPTING_TYPE,
            "status": "success",
            "ttft_ms": "",
            "response_ms": "",
            "total_ms": "",
            "cached_tokens": "",
            "prompt_tokens": "",
            "prefix_hit_rate": "",
            "actual_prefill_worker": "",
            "actual_decode_worker": "",
            "router_decision_worker": "",
            "previous_workers": "-",
            "_request_started_at": None,
            "_response_received_at": None,
            "error": "",
        }
        with metrics_lock:
            row["previous_workers"] = _format_previous_workers(
                prompt_workers.get(prompt, [])
            )
        trace_prefix = _trace_prefix_for_experiment(experiment_name) or [
            "./traces/rr",
            "./traces/kv",
        ]
        trace_request = None
        try:
            measurement = send_request(
                prompt, on_token=on_token, request_id=request_id
            )
            row["_request_started_at"] = measurement["request_started_at"]
            row["_response_received_at"] = measurement["response_received_at"]
            trace_request = latest_request_trace(
                trace_prefix,
                measurement["trace_request_id"],
            )
            if trace_request is None:
                raise RuntimeError(
                    "No request trace found for request ID "
                    f"{measurement['trace_request_id']}"
                )
            trace_metrics = {
                key: trace_request.get(key)
                for key in (
                    "ttft_ms",
                    "total_time_ms",
                    "input_tokens",
                    "cached_tokens",
                )
            }
            if any(
                not isinstance(value, (int, float))
                for value in trace_metrics.values()
            ):
                raise RuntimeError(
                    "Request trace is missing numeric timing or token metrics "
                    f"for request ID {measurement['trace_request_id']}"
                )
            row["ttft_ms"] = trace_metrics["ttft_ms"]
            row["total_ms"] = trace_metrics["total_time_ms"]
            row["response_ms"] = (
                trace_metrics["total_time_ms"] - trace_metrics["ttft_ms"]
            )
            row["cached_tokens"] = trace_metrics["cached_tokens"]
            row["prompt_tokens"] = trace_metrics["input_tokens"]
            request_hit_rate = _request_prefix_hit_rate(
                trace_metrics["cached_tokens"],
                trace_metrics["input_tokens"],
            )
            if request_hit_rate is not None:
                row["prefix_hit_rate"] = f"{request_hit_rate:.2f}"
            response_text = measurement["output"]
            phase = "COMPLETE"
        except Exception as exc:  # noqa: BLE001
            if (
                row["_request_started_at"] is not None
                and row["_response_received_at"] is None
            ):
                row["_response_received_at"] = time.perf_counter()
            row.update(status="failed", error=str(exc))
            response_text = f"Request failed: {exc}"
            phase = "FAILED"
        if row["status"] == "success":
            worker = trace_request.get("worker")
            raw_prefill_worker_id = (
                worker.get("prefill_worker_id")
                if isinstance(worker, dict)
                else ""
            )
            raw_decode_worker_id = (
                worker.get("decode_worker_id")
                if isinstance(worker, dict)
                else ""
            )
            with metrics_lock:
                (
                    row["actual_prefill_worker"],
                    row["actual_decode_worker"],
                    row["previous_workers"],
                ) = _record_prompt_workers(
                    prompt,
                    raw_prefill_worker_id,
                    raw_decode_worker_id,
                    worker_labels,
                    prompt_workers,
                )
                row["router_decision_worker"] = row["actual_decode_worker"]
            if raw_prefill_worker_id not in (None, "") and not row["actual_prefill_worker"]:
                console.print(
                    f"[yellow]Worker ID {raw_prefill_worker_id} was not "
                    "discovered during startup.[/]"
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
        ThreadPoolExecutor(max_workers=CONCURRENCY) as executor,
    ):
        request_pairs = [
            workload[index : index + 2]
            for index in range(0, len(workload), 2)
        ]
        active_requests = {}
        next_pair = 0
        next_request_number = 1

        def number_request(request):
            nonlocal next_request_number
            numbered_request = (
                f"RUN{run_number:02d}-R{next_request_number}",
                *request[1:],
            )
            next_request_number += 1
            return numbered_request

        def submit_priming_request(pair):
            future = executor.submit(
                execute_request,
                *number_request(pair[0]),
            )
            active_requests[future] = (pair, 0)

        while next_pair < len(request_pairs) and len(active_requests) < CONCURRENCY:
            submit_priming_request(request_pairs[next_pair])
            next_pair += 1

        while active_requests or not events.empty():
            completed = [
                future for future in active_requests if future.done()
            ]
            for future in completed:
                pair, request_index = active_requests.pop(future)
                future.result()
                if request_index == 0:
                    test_future = executor.submit(
                        execute_request,
                        *number_request(pair[1]),
                    )
                    active_requests[test_future] = (pair, 1)
                elif next_pair < len(request_pairs):
                    submit_priming_request(request_pairs[next_pair])
                    next_pair += 1

            try:
                request_id, question, phase, text, row, _ = events.get(timeout=0.05)
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

    batch_seconds = _active_request_seconds(results)
    console.print(
        f"Time with requests in flight: {batch_seconds:.2f} s "
        f"({len(workload)} requests, concurrency {CONCURRENCY})"
    )
    return results


def main():
    selections = parse_args()
    console.print("\n[bold cyan]WARMING UP VLLM WORKERS[/]")
    run_warmup()
    worker_labels = {}
    if selections is None:
        selections = choose_experiments()
    if not selections:
        console.print("[dim]No benchmark selected.[/]")
        return
    for experiment_name in selections:
        run_benchmark(experiment_name, worker_labels)


if __name__ == "__main__":
    main()
