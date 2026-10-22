"""Request, measurement, and result calculations for the benchmark."""

import json
import statistics
import time
import urllib.error
import urllib.request

from config import MAX_OUTPUT_TOKENS, MODEL, URL


def format_mean(values):
    if not values:
        return "..."
    return f"{statistics.mean(values):.2f}"


def send_request(prompt, on_token=None):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
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
    try:
        response_context = urllib.request.urlopen(request, timeout=180)
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Request failed with HTTP {exc.code}: {details[:1000]}") from exc

    last_event = None
    with response_context as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            last_event = event
            if event.get("error"):
                raise RuntimeError(f"Model request error: {event['error']}")
            choices = event.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            token_text = (delta.get("reasoning_content") or "") + (
                delta.get("content") or ""
            )
            if token_text:
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                output_parts.append(token_text)
                if on_token is not None:
                    on_token("".join(output_parts))
    end = time.perf_counter()
    if first_token_time is None:
        details = json.dumps(last_event, ensure_ascii=False) if last_event else "No SSE data events received"
        raise RuntimeError(f"No generated tokens received. Last server event: {details[:1000]}")
    return {
        "ttft_ms": (first_token_time - start) * 1000,
        "response_ms": (end - first_token_time) * 1000,
        "total_ms": (end - start) * 1000,
        "output": "".join(output_parts),
    }
