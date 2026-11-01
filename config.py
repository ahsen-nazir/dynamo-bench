"""Benchmark settings."""

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3-0.6B"
REQUESTS = 10
MAX_OUTPUT_TOKENS = 18
OUTPUT_CSV = "roundrobin_prefix_cache_benchmark.csv"
