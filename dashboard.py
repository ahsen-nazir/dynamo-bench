"""NorthCorp Labs branding and Rich benchmark presentation."""

from pathlib import Path

from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import ProgressColumn
from rich.table import Table
from rich.text import Text

from config import CONCURRENCY, MODEL, REQUESTS

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


class LatencyChart:
    """Render total request latency for each request."""

    def __init__(self, results):
        self.results = list(results)

    def __rich_console__(self, console, options):
        rows = successful_requests(self.results)
        if not self.results:
            return
        maximum = max((row["total_ms"] for row in rows), default=0)
        width = max(5, min(60, options.max_width - 87))
        table = Table.grid(padding=(0, 1))
        table.add_column(style=WHITE)
        table.add_column(width=width)
        table.add_column(justify="right", style=WHITE)
        table.add_column(justify="right", style=CYAN)
        table.add_column(width=17, justify="right", style=EMERALD)
        table.add_column(justify="right", style=CYAN)
        table.add_column(justify="right", style=ROYAL_BLUE)
        table.add_column(width=16, justify="right", style=INDIGO)
        table.add_row(
            Text("Request", style=f"bold {WHITE}"),
            Text("Request latency", style=f"bold {WHITE}"),
            Text("Time", style=f"bold {WHITE}"),
            Text("TTFT", style=f"bold {ROYAL_BLUE}"),
            Text("Previous workers", style=f"bold {EMERALD}"),
            Text("Cached tokens", style=f"bold {CYAN}"),
            Text("Prefix Hit Rate (%)", style=f"bold {CYAN}"),
            Text("Router decision", style=f"bold {INDIGO}"),
        )
        for row in self.results:
            if row["status"] != "success":
                table.add_row(
                    row["request_id"],
                    Text("FAILED", style="red"),
                    "—",
                    "—",
                    "—",
                    "—",
                    "—",
                    "—",
                    "—",
                )
                continue
            total = max(1, round(width * row["total_ms"] / maximum)) if maximum else 0
            first = (
                round(total * row["ttft_ms"] / row["total_ms"])
                if row["total_ms"]
                else 0
            )
            bar = Text("█" * first, style=ROYAL_BLUE)
            bar.append("█" * (total - first), style=EMERALD)
            table.add_row(
                row["request_id"],
                bar,
                f"{row['total_ms']:.2f} ms",
                f"{row['ttft_ms']:.2f} ms",
                str(row.get("previous_workers") or "-"),
                str(
                    row["cached_tokens"]
                    if row.get("cached_tokens") not in (None, "")
                    else "—"
                ),
                (
                    f"{float(row['prefix_hit_rate']):.2f}"
                    if row.get("prefix_hit_rate") not in (None, "")
                    else "—"
                ),
                str(
                    row.get("router_decision_worker")
                    if row.get("router_decision_worker") not in (None, "")
                    else "—"
                ),
            )
        yield table
        legend = Text("■ TTFT", style=ROYAL_BLUE)
        legend.append("   ")
        legend.append("■", style=ROYAL_BLUE)
        legend.append("■", style=EMERALD)
        legend.append(" Total", style=MUTED)
        yield legend


def build_dashboard(
    results, progress, request_id, phase, question, response_text, experiment_name
):
    header = Text(f"{request_id}  |  {PHASES.get(phase, phase)}", style=f"bold {WHITE}")
    count = len(successful_requests(results))
    planned_requests = next(
        (task.total for task in progress.tasks if task.total is not None),
        REQUESTS,
    )
    protocol = Text(
        f"RUNxx-R1 -> RUNxx-R{planned_requests} - "
        f"up to {CONCURRENCY} concurrent requests\n",
        style=INDIGO,
    )
    description = "Priming/test requests run in order within each prompt family."
    protocol.append(description, style=MUTED)
    return Panel(
        Group(
            Branding(),
            Text(f"Model: {MODEL}", style=MUTED),
            Panel(protocol, title="Request sequence", border_style=INDIGO),
            header,
            progress,
            Text(
                f"Successful: {count}  |  Failed: {len(results) - count}  |  Planned: {planned_requests}",
                style=EMERALD,
            ),
            Panel(
                LatencyChart(results),
                title="Request latency",
                border_style=EMERALD,
            ),
            Panel(
                Group(
                    Text(
                        question.split("\n\n", 1)[0]
                        if question
                        else "No active request",
                        style=f"bold {CYAN}",
                    ),
                    Text(
                        response_text[-1200:] or "Waiting for streamed output…",
                        style=WHITE,
                    ),
                ),
                title="Streamed model output",
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
                LatencyChart(results),
            ),
            title=f"{benchmark_title(experiment_name)} - final results",
            border_style=EMERALD,
        )
    )
