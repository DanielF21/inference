"""What a batch is made of.

Kept apart from bench.py, which measures, and from prompts.py, which only
holds text. Composition is its own decision and the engines after this one are
graded on workloads built here.

Everything is drawn from a seeded RNG rather than from EOS. Budgets under a
fixed seed are deterministic and controlled here; EOS-driven lengths are a
function of model output, which is one more thing free to move between engines.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from infer.core import Request
from infer.prompts import PROMPTS

# Mixed output lengths, per the load-testing methodology. Held as fractions of
# the run's max_new_tokens rather than fixed counts, so --max-new-tokens is not
# silently ignored and a short smoke run costs seconds instead of minutes.
#
# The mean is 0.625 of the max at any budget, so a static batch discards 37.5%
# of its decode row steps whatever the scale. At 256 this is (64, 128, 192, 256).
BUDGET_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def budgets_for(max_new_tokens: int) -> tuple[int, ...]:
    return tuple(max(1, round(f * max_new_tokens)) for f in BUDGET_FRACTIONS)


def mixed_batch(
    batch_size: int,
    seed: int,
    max_new_tokens: int,
    labels: Sequence[str] | None = None,
    budgets: Sequence[int] | None = None,
) -> list[tuple[str, Request]]:
    """Draw (prompt_label, Request) pairs, ragged on both axes.

    Labels ride alongside the Request because a ragged batch has no single
    prompt label, and the CSV records one per row.

    Draws are independent, so a batch may not span every prompt length. That is
    what an arrival stream looks like, and the padding waste it produces should
    be reported rather than engineered into a fixed composition.
    """
    rng = random.Random(seed)
    pool = list(labels if labels is not None else PROMPTS)
    choices = list(budgets if budgets is not None else budgets_for(max_new_tokens))

    batch = []
    for _ in range(batch_size):
        label = rng.choice(pool)
        batch.append((label, Request(PROMPTS[label], rng.choice(choices))))
    return batch


def poisson_arrivals(count: int, rate: float, seed: int) -> list[float]:
    """Cumulative arrival times in seconds, for `count` requests at `rate`/s.

    Exponential gaps, which is what makes it Poisson. Seeded, so a rate sweep
    can be replayed against a different engine and compared request for
    request.
    """
    rng = random.Random(seed)
    clock = 0.0
    times = []
    for _ in range(count):
        clock += rng.expovariate(rate)
        times.append(clock)
    return times
