"""Aggregate results/runs.csv and compare engines.

All aggregation lives here — nothing pre-aggregated is ever written by the
benchmark, so raw runs stay recoverable.

    uv run python scripts/analyze.py
    uv run python scripts/analyze.py --parity ENGINE_A ENGINE_B
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_rows(include_warmup: bool = False) -> list[dict]:
    """Pool every engine's results/<engine>/runs.csv into one list."""
    paths = sorted(RESULTS_DIR.glob("*/runs.csv"))
    if not paths:
        raise SystemExit(
            f"no results under {RESULTS_DIR}/*/runs.csv — run a benchmark first"
        )

    rows: list[dict] = []
    for path in paths:
        with path.open() as f:
            rows.extend(csv.DictReader(f))
    if not include_warmup:
        rows = [r for r in rows if r["is_warmup"] != "True"]

    # Results recorded before batching predate the column. Each such row is its
    # own batch, which is exactly what it was, so their aggregates are
    # unchanged by everything below.
    for index, row in enumerate(rows):
        if not row.get("batch_id"):
            row["batch_id"] = f"_row{index}"
    return rows


def batch_seconds(group: list[dict], field: str) -> float:
    """Sum `field` counting each batch once.

    Every row in a batch carries the batch's wall clock, so summing rows
    multiplies it by the batch size. That is the difference between reporting
    aggregate throughput and reporting mean per-sequence throughput, which is
    roughly flat in batch size and would make batching look like it did
    nothing.
    """
    per_batch = {r["batch_id"]: float(r[field]) for r in group}
    return sum(per_batch.values())


def summarize(rows: list[dict]) -> None:
    """Per (engine, prompt, batch size) aggregates.

    Throughput uses ratio-of-means (total tokens / total time), never the
    mean of per-run rates — averaging ratios overweights fast runs and
    inflates the number.

    Batch size is part of the key: a sweep pools several batch sizes under one
    prompt label, and averaging across them would report a number describing
    no configuration that was actually run.
    """
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["engine"], r["prompt_label"], int(r["batch_size"]))].append(r)

    header = (
        f"{'engine':<10} {'prompt':<14} {'ptok':>5} {'B':>4} {'n':>3} "
        f"{'ttft_med':>9} {'total_mean':>11} {'decode_tps':>11} {'e2e_tps':>9} "
        f"{'itl_p95':>9}"
    )
    print(header)
    print("-" * len(header))

    ordered = sorted(
        groups.items(),
        key=lambda kv: (int(kv[1][0]["prompt_tokens"]), kv[0][2], kv[0][0]),
    )
    for (engine, prompt, batch_size), group in ordered:
        total_tokens = sum(int(r["completion_tokens"]) for r in group)
        decode_tokens = sum(int(r["completion_tokens"]) - 1 for r in group)
        total_time = batch_seconds(group, "total_s")
        decode_time = batch_seconds(group, "decode_s")

        ttft_med = statistics.median(float(r["ttft_s"]) for r in group)
        itl_p95 = [float(r["itl_p95_ms"]) for r in group if r["itl_p95_ms"]]

        # Mean wall-clock per batch: the denominator for FLOPs-based
        # utilization, which needs a duration rather than a rate.
        n_batches = len({r["batch_id"] for r in group})
        total_mean = total_time / n_batches

        print(
            f"{engine:<10} {prompt:<14} {int(group[0]['prompt_tokens']):>5} "
            f"{batch_size:>4} {len(group):>3} "
            f"{ttft_med:>8.3f}s {total_mean:>10.3f}s {decode_tokens / decode_time:>11.2f} "
            f"{total_tokens / total_time:>9.2f} "
            f"{(statistics.median(itl_p95) if itl_p95 else float('nan')):>8.1f}ms"
        )


def speedup(rows: list[dict], baseline: str) -> None:
    """Aggregate decode-throughput ratio vs the baseline engine at batch 1.

    Columns are engine@batch, because a batched engine's whole claim is that
    the ratio moves with batch size. The baseline is always batch 1: it is the
    thing being improved on, not a configuration to match.
    """
    configs = sorted({(r["engine"], int(r["batch_size"])) for r in rows})
    if (baseline, 1) not in configs or len(configs) < 2:
        return

    def decode_tps(engine: str, batch_size: int, prompt: str) -> float | None:
        group = [r for r in rows
                 if r["engine"] == engine
                 and int(r["batch_size"]) == batch_size
                 and r["prompt_label"] == prompt]
        if not group:
            return None
        tokens = sum(int(r["completion_tokens"]) - 1 for r in group)
        seconds = batch_seconds(group, "decode_s")
        return tokens / seconds if seconds else None

    others = [c for c in configs if c != (baseline, 1)]
    prompts = sorted({r["prompt_label"] for r in rows},
                     key=lambda p: int(next(r["prompt_tokens"] for r in rows if r["prompt_label"] == p)))

    print(f"\naggregate decode_tps speedup vs {baseline}@1")
    print(f"{'prompt':<14} " + " ".join(f"{e + '@' + str(b):>12}" for e, b in others))
    for prompt in prompts:
        base = decode_tps(baseline, 1, prompt)
        cells = []
        for engine, batch_size in others:
            value = decode_tps(engine, batch_size, prompt)
            cells.append(f"{value / base:>11.2f}x" if base and value else f"{'-':>12}")
        print(f"{prompt:<14} " + " ".join(cells))


def padding(rows: list[dict]) -> None:
    """KV slots reserved versus slots that ever hold a token.

    Derived, never recorded. The engine pads a batch to its widest row and
    sizes the cache at width + max_new_tokens, so every quantity here follows
    from columns already present. A recorded field could drift from what the
    engine actually did; a derivation carries its assumption in the open.

    Two kinds of empty slot, kept apart because different engines fix them:

      pad     left padding, a row shorter than the widest in its batch. Only
              exists in a ragged batch. This is what a paged layout reclaims.
      unused  generation budget never spent, because a sequence stopped early.
              Scheduling reclaims this, not layout.

    A uniform batch running to a fixed budget has neither, so both stay at
    0.0% until batches are ragged.
    """
    batches: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        batches[r["batch_id"]].append(r)

    groups: dict[tuple[str, str, int], list[tuple[int, int, int]]] = defaultdict(list)
    for group in batches.values():
        head = group[0]
        # The batch runs to its longest budget and the cache is sized for it,
        # so every row's reservation is the max, not its own.
        budget = max(int(r["max_new_tokens"]) for r in group)
        width = max(int(r["prompt_tokens"]) for r in group)
        reserved = len(group) * (width + budget)
        pad = len(group) * width - sum(int(r["prompt_tokens"]) for r in group)
        unused = len(group) * budget - sum(int(r["completion_tokens"]) for r in group)
        key = (head["engine"], head["prompt_label"], int(head["batch_size"]))
        groups[key].append((reserved, pad, unused))

    header = (
        f"{'engine':<10} {'prompt':<14} {'B':>4} {'reserved':>10} "
        f"{'pad':>9} {'unused':>9} {'pad%':>7} {'unused%':>8}"
    )
    print(f"\nKV slot occupancy (derived)\n{header}")
    print("-" * len(header))

    for (engine, prompt, batch_size), stats in sorted(
        groups.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][1])
    ):
        reserved = sum(s[0] for s in stats)
        pad = sum(s[1] for s in stats)
        unused = sum(s[2] for s in stats)
        print(
            f"{engine:<10} {prompt:<14} {batch_size:>4} {reserved // len(stats):>10} "
            f"{pad // len(stats):>9} {unused // len(stats):>9} "
            f"{100 * pad / reserved:>6.1f}% {100 * unused / reserved:>7.1f}%"
        )


def parity(engine_a: str, engine_b: str) -> None:
    """Report the first index where two engines' greedy token ids diverge.

    First-divergence rather than pass/fail: fp16 with different kernel shapes
    drifts eventually, and where it drifts is the useful signal.
    """
    paths = sorted(RESULTS_DIR.glob("*/runs.jsonl"))
    if not paths:
        raise SystemExit(f"no raw records under {RESULTS_DIR}/*/runs.jsonl")

    latest: dict[tuple[str, str], list[int]] = {}
    for path in paths:
        with path.open() as f:
            for line in f:
                rec = json.loads(line)
                if rec["is_warmup"] or rec["temperature"] not in ("", None):
                    continue  # parity is only meaningful under greedy
                # Batched rows share a prompt label, so a batch would overwrite
                # itself here. Parity is a correctness check and batch 1 is
                # where engines are directly comparable anyway.
                if int(rec.get("batch_size", 1)) != 1:
                    continue
                latest[(rec["engine"], rec["prompt_label"])] = rec["token_ids"]

    prompts = sorted({p for (e, p) in latest if e == engine_a})
    if not prompts:
        raise SystemExit(f"no greedy records for engine {engine_a!r}")

    print(f"{'prompt':<14} {'result':<40}")
    for prompt in prompts:
        a, b = latest.get((engine_a, prompt)), latest.get((engine_b, prompt))
        if a is None or b is None:
            print(f"{prompt:<14} missing record for one engine")
            continue
        diverged = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if diverged is None and len(a) == len(b):
            print(f"{prompt:<14} identical ({len(a)} tokens)")
        elif diverged is None:
            print(f"{prompt:<14} prefix matches, lengths differ ({len(a)} vs {len(b)})")
        else:
            print(f"{prompt:<14} first divergence at token {diverged}/{min(len(a), len(b))}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--include-warmup", action="store_true")
    p.add_argument("--baseline", default="naive")
    p.add_argument("--parity", nargs=2, metavar=("ENGINE_A", "ENGINE_B"))
    args = p.parse_args()

    if args.parity:
        parity(*args.parity)
        return

    rows = load_rows(include_warmup=args.include_warmup)
    summarize(rows)
    speedup(rows, args.baseline)
    padding(rows)


if __name__ == "__main__":
    main()
