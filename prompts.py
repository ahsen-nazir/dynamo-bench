"""Fixed shuffled workload: three prefix groups, four questions per group."""

import random

from config import BASE_PREFIX, SHUFFLE_SEED

GROUPS = (
    "Philosophy: interpret the context through consciousness and personal identity.",
    "Astronomy: interpret the context through cosmic time and civilizations.",
    "Literature: interpret the context through storytelling and human experience.",
    "Science: interpret the context through technical depth.",
    "Art: interpret the context through aesthetic expression.",
)
QUESTIONS = (
    "Summarize the central idea in one sentence.",
    "Identify one assumption and briefly challenge it.",
    "Give one concrete example illustrating the context.",
    "Answer in the light of the context, what is the most important?",
    "Do you understand the context, right from the beginning till then end?",
    "What is the most surprising implication of the context?",
)


def build_requests(count):
    """Return identical prompts and submission order for either routing policy."""
    workload = []
    for group in GROUPS:
        prefix = f"{group}\n\n{BASE_PREFIX}"
        for question in QUESTIONS:
            workload.append((question, f"{prefix}\nQuestion: {question}"))
    if count != len(workload):
        raise ValueError(
            f"This workload requires {len(workload)} requests, got {count}"
        )
    random.Random(SHUFFLE_SEED).shuffle(workload)
    for index, (question, prompt) in enumerate(workload, start=1):
        yield f"R{index}", question, prompt


def build_cache_reuse_requests():
    """Return two requests sharing the same prefix but different questions."""
    prefix = f"{GROUPS[0]}\n\n{BASE_PREFIX}"

    if len(QUESTIONS) < 2:
        raise ValueError("Cache-reuse test requires at least two questions.")

    for index, question in enumerate(QUESTIONS[:2], start=1):
        yield f"R{index}", question, f"{prefix}\nQuestion: {question}"
