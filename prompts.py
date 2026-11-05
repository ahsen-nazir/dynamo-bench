import random

from config import PROMPTING_TYPE, PROMPT_LENGTH, REQUESTS


PROMPT_FAMILIES = REQUESTS // 2

_SHORT_PROMPT_TEMPLATE = """Write a clear briefing about {topic}. Explain the main purpose
and identify the people, places, or systems involved. Describe three important
facts and show how they relate to one another. Include one practical example
that a newcomer can understand without special background knowledge. Point out
a common misconception and correct it using specific reasoning. Mention one
meaningful limitation or tradeoff, and explain what information would help
someone make a better decision. Organize the explanation in a logical order,
use precise everyday language, distinguish evidence from opinion, and finish
with a concise takeaway that follows from the details rather than introducing
an unrelated claim."""

_LONG_PROMPT_TEMPLATE = """Prepare a thorough analytical report about {topic} for readers
who need both a reliable overview and useful detail. Begin by defining the
subject, explaining its context, and identifying the people, institutions,
tools, or natural systems that influence it. Trace how the subject developed
and describe the mechanisms that shape its present operation. Examine several
important examples, including what happened, why it mattered, and which
conditions affected the result. Compare at least two reasonable approaches or
interpretations, stating their assumptions, strengths, weaknesses, and likely
consequences. Explain relevant measurements and distinguish direct evidence
from estimates, uncertainty, and unsupported claims. Address practical costs,
benefits, risks, accessibility, and long-term effects for different groups.
Describe a realistic scenario in which a decision-maker could apply the
analysis, then identify the additional facts that would change the decision.
Use clear headings, define unfamiliar terms, avoid false certainty, and make
the reasoning explicit. Conclude with a balanced summary, a justified
recommendation, and a short list of open questions that deserve further
investigation."""

SHORT_PROMPTS = [
    _SHORT_PROMPT_TEMPLATE.format(topic=topic)
    for topic in (
        "urban transit",
        "river restoration",
        "school gardens",
        "public libraries",
        "coastal mapping",
        "solar cooking",
        "community clinics",
        "wildlife corridors",
        "digital archives",
        "local recycling",
    )
]

LONG_PROMPTS = [
    _LONG_PROMPT_TEMPLATE.format(topic=topic)
    for topic in (
        "renewable electricity",
        "regional food systems",
        "urban heat planning",
        "public health networks",
        "coastal flood resilience",
        "wildlife conservation",
        "responsible data governance",
        "railway modernization",
        "forest restoration",
        "community water management",
    )
]

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


def build_requests(run_number=1):
    """Return priming/test request pairs with prompts unique to this run."""
    if PROMPTING_TYPE == "random":
        prompt_lists = {"short": SHORT_PROMPTS, "long": LONG_PROMPTS}
        try:
            prompts = prompt_lists[PROMPT_LENGTH]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported PROMPT_LENGTH {PROMPT_LENGTH!r}; use 'short' or 'long'"
            ) from exc
    elif PROMPTING_TYPE != "priming":
        raise ValueError(
            f"Unsupported PROMPTING_TYPE {PROMPTING_TYPE!r}; "
            "use 'priming' or 'random'"
        )

    workload = []
    request_number = 1

    for family_index in range(PROMPT_FAMILIES):
        family_id = f"RUN{run_number:02d}-F{family_index + 1:02d}"
        for request_role in ("priming", "test"):
            if PROMPTING_TYPE == "priming":
                prompt = (
                    f"RUN_{run_number:02d}_FAMILY_{family_index:02d}\n\n"
                    f"{SHARED_PROMPT}"
                )
                question = f"Analyze the following information: {prompt}"
            else:
                prompt = random.choice(prompts)
                question = prompt
            workload.append(
                (
                    f"RUN{run_number:02d}-R{request_number}",
                    question,
                    prompt,
                    family_id,
                    request_role,
                )
            )
            request_number += 1

    return workload
