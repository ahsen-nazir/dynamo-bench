"""Paired cold/warm prefix-cache benchmark with a Rich terminal dashboard."""

import csv
import json
import statistics
import time
import urllib.request
import uuid
from functools import partial
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    Progress,
    ProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3-0.6B"
EXPERIMENTS = 10
MAX_OUTPUT_TOKENS = 64
OUTPUT_CSV = "results/roundrobin_prefix_cache_benchmark.csv"

# Reduce repetitions if the complete prompt exceeds the model's context limit.
BASE_PREFIX = (
    "Imagine a universe where consciousness is not an accident, "
    "but a fundamental property of reality. Civilizations rise "
    "and disappear, stars are born and die, and countless minds "
    "search for meaning in an indifferent cosmos. "
    "Some believe that time is an illusion, that every moment "
    "exists simultaneously, and that free will is merely the "
    "feeling of making choices already determined. "
    "Others wonder whether reality itself is a simulation, "
    "whether dreams reveal hidden dimensions of existence, "
    "or whether the universe is somehow aware of itself. "
    "Human beings live for only a brief moment in cosmic history, "
    "yet they create art, fall in love, question their existence, "
    "and imagine worlds they will never see. "
    "Perhaps the greatest mystery is not why the universe exists, "
    "but why anything within it can experience existence at all. "
) * 7

ROSE = "#E5A4B6"
LIGHT_ROSE = "#F2B8C6"
BEIGE = "#EBD2B8"
CREAM = "#F5DFD2"
MUTED = "#C8B4B9"
DARK_ROSE = "#594750"

console = Console()


class ThickBarColumn(ProgressColumn):
    """Render a progress bar using multiple terminal rows."""

    def __init__(self, width=38, height=1):
        super().__init__()
        self.width = width
        self.height = height

    def render(self, task):
        fraction = (task.completed / task.total) if task.total else 0.0
        filled = round(self.width * max(0.0, min(1.0, fraction)))
        bar = Text()
        for row in range(self.height):
            if row:
                bar.append("\n")
            bar.append("█" * filled, style=BEIGE if task.finished else ROSE)
            bar.append("░" * (self.width - filled), style=DARK_ROSE)
        return bar


def mean_sd(values):
    if not values:
        return "..."
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.2f} ± {sd:.2f}"


def metric_values(results, metric):
    cold = [r[f"cold_{metric}_ms"] for r in results]
    warm = [r[f"warm_{metric}_ms"] for r in results]
    saved = [c - w for c, w in zip(cold, warm)]
    reductions = [100 * (1 - w / c) for c, w in zip(cold, warm)]
    return cold, warm, saved, reductions


def build_dashboard(
    results, progress, experiment, phase, question, response_text, failures
):
    header = Text()
    header.append(f"Experiment {experiment}/{EXPERIMENTS}", style=f"bold {CREAM}")
    header.append(f"  |  {phase}", style=LIGHT_ROSE)

    table = Table(
        title="LATENCY (MEAN ± SD)",
        expand=True,
        header_style=f"bold {ROSE}",
        border_style=DARK_ROSE,
    )
    table.add_column("Metric", style=CREAM)
    table.add_column("Cold (ms)", justify="right")
    table.add_column("Warm (ms)", justify="right")
    table.add_column("Saved (ms)", justify="right")

    for label, metric in (("TTFT", "ttft"), ("Total latency", "total")):
        cold, warm, saved, _ = metric_values(results, metric)
        table.add_row(
            label,
            mean_sd(cold),
            f"[{LIGHT_ROSE}]{mean_sd(warm)}[/]",
            f"[{BEIGE}]{mean_sd(saved)}[/]",
        )

    improvements = Table.grid(expand=True)
    improvements.add_column()
    improvements.add_column(justify="right")
    if results:
        for label, metric in (
            ("TTFT reduction", "ttft"),
            ("Total latency reduction", "total"),
        ):
            _, _, _, reductions = metric_values(results, metric)
            improvements.add_row(
                f"[{CREAM}]{label}[/]",
                f"[bold {LIGHT_ROSE}]{mean_sd(reductions)}%[/]",
            )
    improvements.add_row("Successful pairs", f"{len(results)}/{EXPERIMENTS}")
    improvements.add_row("Failed pairs", str(failures))

    phase_color = BEIGE if phase == "COLD" else LIGHT_ROSE
    stream_text = Text(
        response_text[-1200:] or "Waiting for model output...",
        style=CREAM if response_text else MUTED,
    )
    response_panel = Panel(
        Group(
            Text(question or "No active question", style=f"bold {BEIGE}"),
            Text(""),
            stream_text,
        ),
        title=f"[bold {LIGHT_ROSE}]MODEL RESPONSE[/] [bold {phase_color}][ {phase} ][/]",
        border_style=phase_color,
        padding=(1, 2),
    )
    return Panel(
        Group(
            header,
            Text(""),
            progress,
            Text(""),
            table,
            Text(""),
            improvements,
            Text(""),
            response_panel,
        ),
        title=f"[bold {ROSE}]DYNAMO / PREFIX CACHE BENCHMARK[/]",
        border_style=ROSE,
        padding=(1, 2),
    )


def send_request(prompt, on_token=None):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0,
        "stream": True,
    }
    request = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first_token_time = None
    output_parts = []
    with urllib.request.urlopen(request, timeout=180) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            choices = event.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content") or ""
            reasoning = delta.get("reasoning_content") or ""
            token_text = reasoning + content
            if token_text:
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                output_parts.append(token_text)
                if on_token is not None:
                    on_token("".join(output_parts))
    end = time.perf_counter()
    if first_token_time is None:
        raise RuntimeError("No generated tokens received")
    return {
        "ttft_ms": (first_token_time - start) * 1000,
        "response_ms": (end - first_token_time) * 1000,
        "total_ms": (end - start) * 1000,
        "output": "".join(output_parts),
    }


def update_stream(
    text, *, live, results, progress, experiment, phase, question, failures
):
    live.update(
        build_dashboard(results, progress, experiment, phase, question, text, failures),
        refresh=False,
    )


def print_summary(results):
    table = Table(
        title="FINAL BENCHMARK RESULTS (MEAN ± SD)",
        header_style=f"bold {ROSE}",
        border_style=ROSE,
        expand=True,
    )
    table.add_column("Metric", style=CREAM)
    table.add_column("Cold (ms)", justify="right")
    table.add_column("Warm (ms)", justify="right")
    table.add_column("Saved (ms)", justify="right")
    table.add_column("Reduction (%)", justify="right")
    for label, metric in (("TTFT", "ttft"), ("Total latency", "total")):
        cold, warm, saved, reductions = metric_values(results, metric)
        table.add_row(
            label, mean_sd(cold), mean_sd(warm), mean_sd(saved), mean_sd(reductions)
        )
    console.print("\n", table)


def main():
    results = []
    failures = 0
    console.print(f"\n[bold {ROSE}]Initializing Dynamo benchmark...[/]")
    try:
        send_request("Warmup request. Explain AI briefly.")
    except (OSError, RuntimeError, ValueError) as exc:
        console.print(f"[red]Warmup failed: {exc}[/]")
        return

    progress = Progress(
        TextColumn(f"[bold {CREAM}]{{task.description}}[/]"),
        ThickBarColumn(width=38, height=1),
        TextColumn(f"[bold {LIGHT_ROSE}]{{task.percentage:>5.1f}}%[/]"),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=False,
    )
    task = progress.add_task("Requests", total=EXPERIMENTS * 2)

    with Live(
        build_dashboard(results, progress, 0, "READY", "", "", failures),
        console=console,
        refresh_per_second=8,
        transient=False,
        vertical_overflow="visible",
    ) as live:
        for experiment in range(1, EXPERIMENTS + 1):
            # A unique early marker prevents reuse of the long prefix across pairs.
            prefix = f"Experiment identifier: {uuid.uuid4().hex}\n\n{BASE_PREFIX}"
            prompts = [
                (
                    "COLD",
                    "How would a criminal mastermind execute the perfect museum heist in a fictional movie?",
                    prefix
                    + "\nQuestion: "
                    + "How would a criminal mastermind execute the perfect museum heist in a fictional movie?",
                ),
                (
                    "WARM",
                    "What tiny mistake would expose an otherwise brilliant fictional criminal?",
                    prefix
                    + "\nQuestion: "
                    + "What tiny mistake would expose an otherwise brilliant fictional criminal?",
                ),
            ]
            measurements = {}
            for phase, question, prompt in prompts:
                callback = partial(
                    update_stream,
                    live=live,
                    results=results,
                    progress=progress,
                    experiment=experiment,
                    phase=phase,
                    question=question,
                    failures=failures,
                )
                callback("")
                try:
                    measurements[phase] = send_request(prompt, on_token=callback)
                    callback(measurements[phase]["output"])
                except (OSError, RuntimeError, ValueError) as exc:
                    measurements[phase] = None
                    callback(f"Request failed: {exc}")
                finally:
                    progress.advance(task)

            cold = measurements.get("COLD")
            warm = measurements.get("WARM")
            if cold is None or warm is None:
                failures += 1
                live.update(
                    build_dashboard(
                        results,
                        progress,
                        experiment,
                        "FAILED",
                        "Pair incomplete",
                        "",
                        failures,
                    )
                )
                continue

            row = {"experiment": experiment}
            for metric in ("ttft", "response", "total"):
                c = cold[f"{metric}_ms"]
                w = warm[f"{metric}_ms"]
                row[f"cold_{metric}_ms"] = c
                row[f"warm_{metric}_ms"] = w
                row[f"{metric}_saved_ms"] = c - w
                row[f"{metric}_reduction_pct"] = 100 * (1 - w / c)
            results.append(row)
            live.update(
                build_dashboard(
                    results,
                    progress,
                    experiment,
                    "COMPLETE",
                    "Experiment completed",
                    warm["output"],
                    failures,
                )
            )

    if not results:
        console.print("[red]No successful pairs.[/]")
        return
    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print_summary(results)
    console.print(
        f"\n[bold {BEIGE}]Saved {len(results)} paired experiments to {output_path}[/]"
    )


if __name__ == "__main__":
    main()
