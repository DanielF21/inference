"""Open-loop arrival simulation: requests show up over time and wait.

A closed test, where every request exists at t=0, makes static batching look
strictly good. Throughput rises with batch size and per-request latency
improves because nothing queues. Its defect only appears when requests arrive
while a batch is already running, because a static batch cannot admit them: a
request that misses by a millisecond waits for the whole batch to finish.

That wait is the number iteration-level scheduling exists to remove, and it has
to be measured here, on the engine that has the problem, before the engine that
fixes it exists.

Time is virtual, not slept. The clock advances by each batch's *measured*
duration, and jumps forward when the queue is empty. So the GPU work is real
and the queueing is exact, while the run costs only what the batches cost
rather than the length of the arrival stream. Sleeping through a 400-second
arrival stream would measure the same thing and add timer jitter.
"""

from __future__ import annotations

import csv
import json
import statistics
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infer.core import PoolExhausted, Request, SamplingConfig

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

ARRIVAL_FIELDS = [
    "sweep_id", "ts_utc", "engine", "engine_params", "device_name",
    "rate_per_s", "max_batch", "seed", "n_requests", "n_unserved",
    "batch_index", "prompt_label", "prompt_tokens", "max_new_tokens",
    "completion_tokens", "arrival_s", "start_s", "queue_s", "ttft_s", "latency_s",
]


@dataclass
class ArrivalRecord:
    """One request's life, timed from when it arrived rather than when it ran.

    Measuring TTFT from batch start is the common mistake here: it reports a
    number the client never experiences and hides the entire queueing effect.
    """

    batch_index: int
    prompt_label: str
    prompt_tokens: int
    max_new_tokens: int
    completion_tokens: int
    arrival_s: float
    start_s: float
    queue_s: float    # start - arrival: how long it sat waiting
    ttft_s: float     # first token, from arrival
    latency_s: float  # completion, from arrival


def replay(
    engine: Any,
    batch: Sequence[tuple[str, Request]],
    arrivals: Sequence[float],
    cfg: SamplingConfig,
    *,
    max_batch: int,
) -> tuple[list[ArrivalRecord], int]:
    """Serve `batch` against `arrivals` under a strictly static policy.

    Form a batch from whatever has arrived, run it to completion, and only then
    look at the queue again. No mid-batch admission, no eviction. The dumbness
    is the measurement: adding admission control here would quietly be
    continuous batching and would delete the number this exists to produce.

    Returns (records, unserved). A pool refusal stops admission rather than
    discarding the run: everything already served is real and worth keeping,
    and how far it got before refusing is itself the finding.
    """
    if len(batch) != len(arrivals):
        raise ValueError(f"{len(batch)} requests against {len(arrivals)} arrivals")

    clock = 0.0
    index = 0
    batch_index = 0
    records: list[ArrivalRecord] = []

    while index < len(batch):
        # Idle: nothing has arrived yet, so skip ahead rather than spin.
        clock = max(clock, arrivals[index])

        window: list[int] = []
        while (
            index < len(batch)
            and arrivals[index] <= clock
            and len(window) < max_batch
        ):
            window.append(index)
            index += 1

        requests = [batch[i][1] for i in window]
        start = clock
        try:
            results = engine.generate_batch(requests, cfg)
        except PoolExhausted as exc:
            # Every later batch would be refused the same way, so stop here and
            # report what was served rather than losing it.
            print(f"[arrivals]   refused after {len(records)} served: {exc}")
            return records, len(batch) - len(records)

        # Every row in a batch shares its prefill and its step boundaries.
        duration = results[0].total_s
        first_token = results[0].ttft_s

        for slot, result in zip(window, results):
            arrival = arrivals[slot]
            records.append(ArrivalRecord(
                batch_index=batch_index,
                prompt_label=batch[slot][0],
                prompt_tokens=result.prompt_tokens,
                max_new_tokens=batch[slot][1].max_new_tokens,
                completion_tokens=result.completion_tokens,
                arrival_s=arrival,
                start_s=start,
                queue_s=start - arrival,
                ttft_s=(start + first_token) - arrival,
                latency_s=(start + duration) - arrival,
            ))

        clock = start + duration
        batch_index += 1

    return records, 0


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank. Matches bench.py so the two are directly comparable."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, round(q / 100 * len(ordered) + 0.5) - 1))
    return ordered[idx]


def summarize(records: list[ArrivalRecord], rate: float) -> dict[str, float]:
    """Queueing and latency percentiles for one arrival rate.

    Throughput is served requests over the span from the first arrival to the
    last completion, so idle time waiting for arrivals counts against it. That
    is what an open system's throughput means.
    """
    if not records:
        return {"rate": rate, "served": 0}

    queue = [r.queue_s for r in records]
    ttft = [r.ttft_s for r in records]
    latency = [r.latency_s for r in records]
    span = max(r.arrival_s + r.latency_s for r in records)

    return {
        "rate": rate,
        "served": len(records),
        "served_per_s": len(records) / span if span else float("nan"),
        "tokens_per_s": sum(r.completion_tokens for r in records) / span if span else float("nan"),
        "queue_p50": statistics.median(queue),
        "queue_p95": percentile(queue, 95),
        "queue_p99": percentile(queue, 99),
        "ttft_p50": statistics.median(ttft),
        "ttft_p95": percentile(ttft, 95),
        "ttft_p99": percentile(ttft, 99),
        "latency_p50": statistics.median(latency),
        "latency_p95": percentile(latency, 95),
    }


def print_summary(rows: list[dict[str, float]]) -> None:
    header = (
        f"{'rate/s':>7} {'served':>7} {'served/s':>9} {'tok/s':>8} "
        f"{'queue_p50':>10} {'queue_p95':>10} {'queue_p99':>10} "
        f"{'ttft_p95':>9} {'lat_p95':>9}"
    )
    print(f"\n{header}\n{'-' * len(header)}")
    for r in rows:
        if not r.get("served"):
            print(f"{r['rate']:>7.1f} {'0':>7}   (nothing served)")
            continue
        print(
            f"{r['rate']:>7.1f} {r['served']:>7} {r['served_per_s']:>9.2f} "
            f"{r['tokens_per_s']:>8.1f} "
            f"{r['queue_p50']:>9.2f}s {r['queue_p95']:>9.2f}s {r['queue_p99']:>9.2f}s "
            f"{r['ttft_p95']:>8.2f}s {r['latency_p95']:>8.2f}s"
        )


def write_arrivals(engine_name: str, rows: list[dict[str, Any]]) -> Path:
    """Append to results/<engine>/arrivals.csv.

    A separate file from runs.csv: the unit there is a run of one engine
    configuration, here it is one request's journey through a queue. Forcing
    them into one schema would leave most columns empty in both directions.

    Takes rows rather than an engine, so a remote worker can hand records back
    for the local side to write, the way bench.write_results already does.
    """
    out_dir = RESULTS_DIR / engine_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "arrivals.csv"

    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ARRIVAL_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return path


def run_sweep(
    engine: Any,
    rates: Sequence[float],
    *,
    cfg: SamplingConfig,
    n_requests: int,
    max_batch: int,
    seed: int,
    device_name: str = "",
    write: bool = True,
) -> tuple[list[dict[str, float]], list[dict[str, Any]]]:
    """One arrival trace per rate. Returns (summaries, csv-ready rows).

    The trace is drawn from the same seed at every rate and for every engine,
    so comparing scheduling policies later means the same requests arriving at
    the same moments rather than two samples of one distribution.
    """
    from infer.workload import mixed_batch, poisson_arrivals

    sweep_id = uuid.uuid4().hex[:8]
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    params = json.dumps(engine.describe(), sort_keys=True)

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, float]] = []

    for rate in rates:
        batch = mixed_batch(n_requests, seed=seed,
                            max_new_tokens=cfg.max_new_tokens)
        arrivals = poisson_arrivals(n_requests, rate=rate, seed=seed)
        print(f"[arrivals] rate={rate}/s n={n_requests} max_batch={max_batch}")

        records, unserved = replay(engine, batch, arrivals, cfg, max_batch=max_batch)

        for record in records:
            rows.append({
                "sweep_id": sweep_id, "ts_utc": stamp,
                "engine": engine.name, "engine_params": params,
                "device_name": device_name,
                "rate_per_s": rate, "max_batch": max_batch,
                "seed": seed, "n_requests": n_requests, "n_unserved": unserved,
                **asdict(record),
            })

        summary = summarize(records, rate)
        summaries.append(summary)
        if records:
            print(f"[arrivals]   {len(records)} served"
                  f"{f', {unserved} unserved' if unserved else ''}, "
                  f"queue p95 {summary['queue_p95']:.2f}s, "
                  f"ttft p95 {summary['ttft_p95']:.2f}s")

    if write and rows:
        path = write_arrivals(engine.name, rows)
        print(f"[arrivals] wrote {len(rows)} records -> {path}")
    return summaries, rows
