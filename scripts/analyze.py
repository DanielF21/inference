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
    return rows


def summarize(rows: list[dict]) -> None:
    """Per (engine, prompt) aggregates.

    Throughput uses ratio-of-means (total tokens / total time), never the
    mean of per-run rates — averaging ratios overweights fast runs and
    inflates the number.
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["engine"], r["prompt_label"])].append(r)

    header = (
        f"{'engine':<10} {'prompt':<14} {'ptok':>5} {'n':>3} "
        f"{'ttft_med':>9} {'total_mean':>11} {'decode_tps':>11} {'e2e_tps':>9} "
        f"{'itl_p95':>9}"
    )
    print(header)
    print("-" * len(header))

    for (engine, prompt), group in sorted(groups.items(), key=lambda kv: int(kv[1][0]["prompt_tokens"])):
        total_tokens = sum(int(r["completion_tokens"]) for r in group)
        total_time = sum(float(r["total_s"]) for r in group)
        decode_tokens = sum(int(r["completion_tokens"]) - 1 for r in group)
        decode_time = sum(float(r["decode_s"]) for r in group)

        ttft_med = statistics.median(float(r["ttft_s"]) for r in group)
        itl_p95 = [float(r["itl_p95_ms"]) for r in group if r["itl_p95_ms"]]

        # Mean wall-clock per run: the denominator for FLOPs-based utilization,
        # which needs a per-run duration rather than a rate.
        total_mean = total_time / len(group)

        print(
            f"{engine:<10} {prompt:<14} {int(group[0]['prompt_tokens']):>5} {len(group):>3} "
            f"{ttft_med:>8.3f}s {total_mean:>10.3f}s {decode_tokens / decode_time:>11.2f} "
            f"{total_tokens / total_time:>9.2f} "
            f"{(statistics.median(itl_p95) if itl_p95 else float('nan')):>8.1f}ms"
        )


def speedup(rows: list[dict], baseline: str) -> None:
    """Cross-engine decode-throughput ratio vs the baseline engine."""
    engines = sorted({r["engine"] for r in rows})
    if baseline not in engines or len(engines) < 2:
        return

    def decode_tps(engine: str, prompt: str) -> float | None:
        group = [r for r in rows if r["engine"] == engine and r["prompt_label"] == prompt]
        if not group:
            return None
        tokens = sum(int(r["completion_tokens"]) - 1 for r in group)
        seconds = sum(float(r["decode_s"]) for r in group)
        return tokens / seconds if seconds else None

    others = [e for e in engines if e != baseline]
    prompts = sorted({r["prompt_label"] for r in rows},
                     key=lambda p: int(next(r["prompt_tokens"] for r in rows if r["prompt_label"] == p)))

    print(f"\ndecode_tps speedup vs {baseline!r}")
    print(f"{'prompt':<14} " + " ".join(f"{e:>12}" for e in others))
    for prompt in prompts:
        base = decode_tps(baseline, prompt)
        cells = []
        for engine in others:
            value = decode_tps(engine, prompt)
            cells.append(f"{value / base:>11.2f}x" if base and value else f"{'—':>12}")
        print(f"{prompt:<14} " + " ".join(cells))


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


if __name__ == "__main__":
    main()
