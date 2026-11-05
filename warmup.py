"""Warm up each vLLM worker with a short UUID prompt."""

import http.client
import json
import urllib.error
import urllib.request
import uuid

from config import MODEL, WARMUP_MAX_OUTPUT_TOKENS, WORKER_PORTS


def build_warmup_request(port, warmup_id):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": warmup_id}],
        "max_tokens": WARMUP_MAX_OUTPUT_TOKENS,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    return urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def warm_worker(worker_id, port, warmup_id):
    request = build_warmup_request(port, warmup_id)
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{worker_id} on port {port} returned HTTP {exc.code}: "
            f"{details[:1000]}"
        ) from exc
    except UnicodeError as exc:
        raise RuntimeError(
            f"{worker_id} on port {port} returned a non-UTF-8 response"
        ) from exc
    except (
        urllib.error.URLError,
        http.client.HTTPException,
        TimeoutError,
        OSError,
    ) as exc:
        raise RuntimeError(
            f"Could not warm {worker_id} on port {port}: {exc}"
        ) from exc


def run_warmup():
    warmup_id = str(uuid.uuid4())
    print(f"Warming workers with UUID {warmup_id}")
    print("Configured worker ports:")
    for worker_id, port in WORKER_PORTS.items():
        print(f"  {worker_id}: {port}")
    failures = []

    for worker_id, port in WORKER_PORTS.items():
        try:
            warm_worker(worker_id, port, warmup_id)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            failures.append(worker_id)
        else:
            print(f"Warmed {worker_id} on port {port}")

    if failures:
        raise RuntimeError(f"Warm-up failed for: {', '.join(failures)}")
    print("All workers warmed successfully.")
    return None


def main():
    run_warmup()


if __name__ == "__main__":
    main()
