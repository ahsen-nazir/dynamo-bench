"""NorthCorp Labs branding and Rich benchmark presentation."""

from pathlib import Path

from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import ProgressColumn
from rich.table import Table
from rich.text import Text

from config import REQUESTS, CONCURRENCY, MODEL
from logic import format_mean

ROYAL_BLUE = "#6C8CFF"
EMERALD = "#40D9A0"
CYAN = "#08C2DC"
WHITE = "#EAF0FF"
MUTED = "#9BAAC5"
INDIGO = "#A497FF"
TRACK = "#283653"

console = Console()
LOGO = Path(__file__).with_name("logo.txt").read_text(encoding="utf-8").rstrip()
NAMES = {"kv-router": "KV Router", "round-robin": "Round Robin"}
PHASES = {
    "READY": "Awaiting first request",
    "RUNNING": "Streaming response",
    "FAILED": "Request failed",
    "COMPLETE": "Request complete",
}
# Five-row lettering keeps the banner portable across Unicode terminals.
LETTERS = {
    "N": ["█   █", "██  █", "█ █ █", "█  ██", "█   █"],
    "O": [" ███ ", "█   █", "█   █", "█   █", " ███ "],
    "R": ["████ ", "█   █", "████ ", "█  █ ", "█   █"],
    "T": ["█████", "  █  ", "  █  ", "  █  ", "  █  "],
    "H": ["█   █", "█   █", "█████", "█   █", "█   █"],
    "C": [" ████", "█    ", "█    ", "█    ", " ████"],
    "P": ["████ ", "█   █", "████ ", "█    ", "█    "],
    "L": ["█    ", "█    ", "█    ", "█    ", "█████"],
    "A": [" ███ ", "█   █", "█████", "█   █", "█   █"],
    "B": ["████ ", "█   █", "████ ", "█   █", "████ "],
    "S": [" ████", "█    ", " ███ ", "    █", "████ "],
}


def benchmark_title(name):
    return f"{NAMES.get(name, name)} benchmark"


def _lettering(word):
    return "\n".join(
        " ".join(LETTERS[letter][row] for letter in word) for row in range(5)
    )


class Branding:
    """Use large lettering where space allows, with a compact fallback."""

    def __rich_console__(self, console, options):
        wordmark = (
            Group(
                Text(_lettering("NORTHCORP"), style=f"bold {WHITE}"),
                Text(_lettering("LABS"), style=f"bold {CYAN}"),
            )
            if options.max_width >= 80
            else Text("NorthCorp Labs", style=f"bold {WHITE}")
        )
        if options.max_width < 48:
            yield Text(LOGO, style=CYAN)
            yield wordmark
            return
        grid = Table.grid(padding=(0, 2))
        grid.add_column(width=22)
        grid.add_column()
        grid.add_row(Text(LOGO, style=CYAN), wordmark)
        yield grid


class ThickBarColumn(ProgressColumn):
    """Render request completion using a blue-to-green progress bar."""

    def __init__(self, width=38, height=1):
        super().__init__()
        self.width = width
        self.height = height

    def render(self, task):
        fraction = task.completed / task.total if task.total else 0.0
        filled = round(self.width * max(0.0, min(1.0, fraction)))
        bar = Text()
        for row in range(self.height):
            if row:
                bar.append("\n")
            bar.append("█" * filled, style=EMERALD if task.finished else ROYAL_BLUE)
            bar.append("░" * (self.width - filled), style=TRACK)
        return bar


def successful_requests(results):
    return [row for row in results if row["status"] == "success"]


METRICS = (
    ("Time to first token (TTFT)", "ttft_ms"),
    ("First token -> stream end", "response_ms"),
    ("End-to-end request latency", "total_ms"),
)


def latency_table(results):
    rows = successful_requests(results)
    table = Table(expand=True, header_style=f"bold {ROYAL_BLUE}", border_style=TRACK)
    table.add_column("Metric (ms)", style=WHITE)
    table.add_column("Mean", justify="right", style=CYAN)
    table.add_column("Min", justify="right", style=EMERALD)
    table.add_column("Max", justify="right", style=INDIGO)
    for label, metric in METRICS:
        values = [row[metric] for row in rows]
        table.add_row(
            label,
            format_mean(values),
            f"{min(values):.2f}" if values else "—",
            f"{max(values):.2f}" if values else "—",
        )
    return table


class LatencyChart:
    """Stack TTFT and post-first-token time on a common millisecond scale."""

    def __init__(self, results):
        self.results = list(results)

    def __rich_console__(self, console, options):
        rows = successful_requests(self.results)
        if not self.results:
            yield Text("Bars appear as requests complete.", style=MUTED)
            return
        maximum = max((row["total_ms"] for row in rows), default=0)
        width = max(5, min(60, options.max_width - 32))
        table = Table.grid(padding=(0, 1))
        table.add_column(style=WHITE)
        table.add_column(width=width)
        table.add_column(justify="right", style=WHITE)
        for row in self.results:
            if row["status"] != "success":
                table.add_row(row["request_id"], Text("FAILED", style="red"), "—")
                continue
            total = max(1, round(width * row["total_ms"] / maximum)) if maximum else 0
            first = (
                round(total * row["ttft_ms"] / row["total_ms"])
                if row["total_ms"]
                else 0
            )
            bar = Text("█" * first, style=ROYAL_BLUE)
            bar.append("█" * (total - first), style=EMERALD)
            table.add_row(row["request_id"], bar, f"{row['total_ms']:.2f} ms")
        yield table
        legend = Text("■ TTFT", style=ROYAL_BLUE)
        legend.append("   ■ After first token", style=EMERALD)
        yield legend
        yield Text(
            f"Common scale: 0–{maximum:.2f} ms. Bar length = total latency.",
            style=MUTED,
        )


def rankings(results):
    rows = successful_requests(results)
    if not rows:
        return Text("No successful requests to rank.", style=MUTED)
    table = Table(expand=True, border_style=TRACK, header_style=EMERALD)
    table.add_column("Metric")
    table.add_column("Least time", style=EMERALD)
    table.add_column("Most time", style=INDIGO)
    for label, metric in METRICS:
        low = min(row[metric] for row in rows)
        high = max(row[metric] for row in rows)
        fastest = ", ".join(row["request_id"] for row in rows if row[metric] == low)
        slowest = ", ".join(row["request_id"] for row in rows if row[metric] == high)
        table.add_row(label, f"{fastest} - {low:.2f} ms", f"{slowest} - {high:.2f} ms")
    return table


INTERPRETATION = (
    "High TTFT can indicate prefill, queueing, or transport delay. "
    "Low TTFT with a large green segment suggests most time was spent after the first token "
    "(including decoding and transport). Total latency alone does not identify prefill or cache hits."
)


def build_dashboard(
    results, progress, request_id, phase, question, response_text, experiment_name
):
    header = Text(f"{request_id}  |  {PHASES.get(phase, phase)}", style=f"bold {WHITE}")
    count = len(successful_requests(results))
    protocol = Text(
        f"R1 -> R{REQUESTS} - fixed shuffled order - concurrency {CONCURRENCY}\n",
        style=INDIGO,
    )
    protocol.append(
        "Three prefix groups, four different questions per group. Every request is measured independently.",
        style=MUTED,
    )
    return Panel(
        Group(
            Branding(),
            Text(f"Model: {MODEL}", style=MUTED),
            Panel(protocol, title="Request sequence", border_style=INDIGO),
            header,
            progress,
            Text(
                f"Successful: {count}  |  Failed: {len(results) - count}  |  Planned: {REQUESTS}",
                style=EMERALD,
            ),
            Panel(
                Group(
                    latency_table(results),
                    Text(
                        "Statistics use successful requests only. TTFT includes reasoning output when present.",
                        style=MUTED,
                    ),
                ),
                title="Latency statistics - milliseconds",
                border_style=ROYAL_BLUE,
            ),
            Panel(
                Group(LatencyChart(results), Text(INTERPRETATION, style=MUTED)),
                title="Per-request latency",
                border_style=EMERALD,
            ),
            Panel(
                Group(
                    Text(question or "No active request", style=f"bold {CYAN}"),
                    Text(
                        response_text[-1200:] or "Waiting for streamed output…",
                        style=WHITE,
                    ),
                ),
                title="Streamed model output - latest 1,200 characters",
                border_style=CYAN,
            ),
        ),
        title=Text(benchmark_title(experiment_name), style=f"bold {WHITE}"),
        border_style=ROYAL_BLUE,
        padding=(0, 1),
    )


def update_stream(
    text, *, live, results, progress, request_id, phase, question, experiment_name
):
    live.update(
        build_dashboard(
            results, progress, request_id, phase, question, text, experiment_name
        ),
        refresh=False,
    )


def print_summary(results, experiment_name):
    count = len(successful_requests(results))
    console.print(
        Panel(
            Group(
                Text(
                    f"Successful: {count} - Failed: {len(results) - count}", style=WHITE
                ),
                latency_table(results),
                rankings(results),
                LatencyChart(results),
                Text(INTERPRETATION, style=MUTED),
            ),
            title=f"{benchmark_title(experiment_name)} - final results",
            border_style=EMERALD,
        )
    )
