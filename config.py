"""Benchmark settings and reusable prompt text."""

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3-0.6B"
REQUESTS = 30
CONCURRENCY = 2
SHUFFLE_SEED = 42
MAX_OUTPUT_TOKENS = 32
OUTPUT_CSV = "roundrobin_prefix_cache_benchmark.csv"

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
) * 2
