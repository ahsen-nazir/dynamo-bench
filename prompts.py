"""Reproducible paired prompt-affinity workload."""

PROMPT_FAMILIES = 10

SHARED_PROMPT = """Analyze a distributed language model serving system and explain how
requests move from an API frontend through a router to independent inference
workers. Describe how a transformer processes input tokens during prefill,
creates key and value tensors in each attention layer, and stores those tensors
in fixed-size KV cache blocks. Explain how prefix caching identifies reusable
blocks, why matching token sequences must begin at the same position, and how
reuse can reduce prefill computation and time to first token. Discuss the
tradeoffs among cache capacity, memory pressure, eviction, worker load, and
request latency. Compare round-robin routing with a cache-aware policy that
considers which worker already contains a request's prefix, while explaining
why either policy may send a request to a different worker. Describe how a
benchmark can distinguish cached prompt tokens from cumulative cache hit and
query counters, and why per-request hit rates should be calculated from
counter differences sampled on the actual prefill worker. Include how to keep
model settings, prompt content, output limits, and request ordering consistent
when comparing routing strategies. Explain why repeated requests with exactly
the same tokenized prompt are useful for testing affinity, why each independent
family should begin with a distinct prefix, and how a priming request followed
by a test request can reveal whether routing reused the worker's cached blocks.
Finally, outline how to record request identity, family, request role, selected
prefill worker, cached tokens, time to first token, total latency, and counter
snapshots so that results can be interpreted without confusing routing history
with the current request's measurements."""


def build_requests():
    """Return two sequential, identical requests for each unique prompt family."""
    workload = []
    request_number = 1

    for family_index in range(PROMPT_FAMILIES):
        family_id = f"F{family_index + 1:02d}"
        prompt = f"FAMILY_{family_index:02d}\n\n{SHARED_PROMPT}"
        question = f"Analyze the following information: {prompt}"

        for request_role in ("priming", "test"):
            workload.append(
                (
                    f"R{request_number}",
                    question,
                    prompt,
                    family_id,
                    request_role,
                )
            )
            request_number += 1

    return workload
