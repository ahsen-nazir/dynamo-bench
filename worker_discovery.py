"""Read request traces to identify the worker selected for each request."""

import gzip
import json
import re
import time
from pathlib import Path


def _trace_files(trace_prefix):
    prefix_path = Path(trace_prefix)
    pattern = re.compile(rf"{re.escape(prefix_path.name)}\.(\d+)\.jsonl\.gz")
    return sorted(
        (
            (int(match.group(1)), path)
            for path in prefix_path.parent.glob(
                f"{prefix_path.name}.*.jsonl.gz"
            )
            if (match := pattern.fullmatch(path.name))
        ),
        reverse=True,
    )


def request_id_from_completion_id(completion_id):
    """Convert a model completion ID to the ID written in request traces."""
    if not isinstance(completion_id, str):
        raise ValueError(f"Invalid model completion ID: {completion_id!r}")
    prefix, separator, request_id = completion_id.partition("-")
    if not separator or prefix not in {"cmpl", "chatcmpl"}:
        raise ValueError(f"Invalid model completion ID: {completion_id!r}")
    if not request_id:
        raise ValueError("Model completion ID is missing its request ID")
    return request_id


def _latest_request_end(trace_prefix, request_id):
    for _, path in _trace_files(trace_prefix):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as trace_file:
                for line in trace_file:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    event = record.get("event") if isinstance(record, dict) else None
                    request = event.get("request") if isinstance(event, dict) else None
                    if (
                        isinstance(event, dict)
                        and event.get("event_type") == "request_end"
                        and isinstance(request, dict)
                        and str(request.get("request_id")) == request_id
                    ):
                        return request
        except (OSError, EOFError, UnicodeError):
            continue
    return None


def latest_request_trace(trace_prefix, request_id, timeout=10.0):
    """Wait for an exact trace request ID match and return its request."""
    trace_prefixes = (
        trace_prefix
        if isinstance(trace_prefix, (list, tuple))
        else (trace_prefix,)
    )
    deadline = time.monotonic() + timeout
    while True:
        for prefix in trace_prefixes:
            request = _latest_request_end(prefix, request_id)
            if request is not None:
                worker = request.get("worker")
                if not isinstance(worker, dict):
                    raise RuntimeError(
                        f"Trace for request {request_id} has no worker IDs"
                    )
                prefill_worker_id = worker.get("prefill_worker_id")
                decode_worker_id = worker.get("decode_worker_id")
                if prefill_worker_id in (None, "") or decode_worker_id in (
                    None,
                    "",
                ):
                    raise RuntimeError(
                        f"Trace for request {request_id} has incomplete "
                        "worker IDs"
                    )
                if prefill_worker_id != decode_worker_id:
                    raise RuntimeError(
                        "Prefill and decode worker IDs differ: "
                        f"prefill={prefill_worker_id}, "
                        f"decode={decode_worker_id}"
                    )
                return request
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.05)
