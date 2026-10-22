"""Run concurrent requests against an independently started frontend."""

import csv
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

from rich.live import Live
from rich.progress import Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from config import REQUESTS, CONCURRENCY, OUTPUT_CSV
from dashboard import (
    EMERALD,
    WHITE,
    CYAN,
    ROYAL_BLUE,
    ThickBarColumn,
    build_dashboard,
    console,
    print_summary,
    benchmark_title,
)
from logic import send_request
from prompts import build_requests, build_cache_reuse_requests


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the request latency benchmark.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--cache-reuse",
        action="store_true",
        help="Run only the two-request cache-reuse test",
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


def run_benchmark(experiment_name):
    results = []
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
    task = progress.add_task(
        "Requests processed",
        total=2 if experiment_name == "cache-reuse" else REQUESTS,
    )
    if experiment_name == "cache-reuse":
        workload = list(build_cache_reuse_requests())
        if len(workload) != 2:
            raise ValueError("Cache-reuse test must generate exactly two requests.")
    else:
        workload = list(build_requests(REQUESTS))
    events = Queue()

    def execute_request(request_id, question, prompt):
        events.put((request_id, question, "RUNNING", "", None, None))

        def on_token(text):
            events.put((request_id, question, "RUNNING", text, None, None))

        row = {
            "request_id": request_id,
            "status": "success",
            "ttft_ms": "",
            "response_ms": "",
            "total_ms": "",
            "error": "",
        }
        try:
            measurement = send_request(prompt, on_token=on_token)
            for metric in ("ttft_ms", "response_ms", "total_ms"):
                row[metric] = measurement[metric]
            response_text = measurement["output"]
            phase = "COMPLETE"
        except Exception as exc:  # noqa: BLE001
            row.update(status="failed", error=str(exc))
            response_text = f"Request failed: {exc}"
            phase = "FAILED"
        finished_at = time.perf_counter()
        events.put((request_id, question, phase, response_text, row, finished_at))

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
        started_at = time.perf_counter()
        for request in workload:
            if experiment_name == "cache-reuse":
                for request in workload:
                    executor.submit(execute_request, *request).result()
            else:
                for request in workload:
                    executor.submit(execute_request, *request)
        finished_at = started_at
        while len(results) < len(workload):
            request_id, question, phase, text, row, completed_at = events.get()
            if row is not None:
                results.append(row)
                results.sort(key=lambda result: int(result["request_id"][1:]))
                finished_at = max(finished_at, completed_at)
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

    batch_seconds = finished_at - started_at
    console.print(
        f"Batch completion time: {batch_seconds:.2f} s ({REQUESTS} requests, concurrency {CONCURRENCY})"
    )

    if results:
        with open(output_csv, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        console.print(
            f"\n[bold {EMERALD}]Saved {len(results)} request records to {output_csv}[/]"
        )
    print_summary(results, experiment_name)


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
