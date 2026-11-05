"""Benchmark settings."""

import os


URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3-0.6B"
MAX_OUTPUT_TOKENS = 18
WARMUP_MAX_OUTPUT_TOKENS = 5
OUTPUT_CSV = "roundrobin_prefix_cache_benchmark.csv"
REQUESTS = 10
CONCURRENCY = 1
RUNS = 10
PROMPTING_TYPE = "random"  # "priming" or "random"
PROMPT_LENGTH = "long"
WORKER_PORTS = {
    f"w{index - 1}": int(os.environ.get(f"VLLM_WORKER{index}_HTTP_PORT", default_port))
    for index, default_port in enumerate((8100, 8110, 8120, 8130), start=1)
}
