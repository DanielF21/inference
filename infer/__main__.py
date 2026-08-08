"""CLI: benchmark one engine per invocation.

    uv run python -m infer --engine naive --runs 5 --warmup 2

Engines are compared by running each separately and appending to the same
results/runs.csv, not by loading several at once.
"""

from __future__ import annotations

import argparse

from infer.core import EngineConfig, SamplingConfig
from infer.engines import ENGINE_NAMES
from infer.prompts import PROMPTS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="infer", description=__doc__.split("\n")[0])
    p.add_argument("--engine", default="naive", choices=ENGINE_NAMES)
    p.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--runs", type=int, default=5, help="recorded runs per prompt")
    p.add_argument("--warmup", type=int, default=2, help="discarded-from-analysis runs")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--prompts", default="", help="comma-separated labels (default: all)"
    )
    # Sampling is off by default: benchmarks run greedy so repeats do identical
    # work and engines can be compared token-for-token.
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--min-p", type=float, default=None)
    p.add_argument("--stop-on-eos", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.prompts:
        labels = [s.strip() for s in args.prompts.split(",") if s.strip()]
        unknown = [label for label in labels if label not in PROMPTS]
        if unknown:
            raise SystemExit(f"unknown prompt label(s): {unknown}; have {list(PROMPTS)}")
        prompts = {label: PROMPTS[label] for label in labels}
    else:
        prompts = dict(PROMPTS)

    sampling = SamplingConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        seed=args.seed,
        stop_on_eos=args.stop_on_eos,
    )

    # Imported here, not at module scope, so --help never loads a model.
    from infer.bench import run_benchmark
    from infer.engines import build_engine

    engine = build_engine(args.engine, EngineConfig(device=args.device, dtype=args.dtype))
    run_benchmark(
        engine, prompts, sampling,
        device=args.device, dtype=args.dtype,
        runs=args.runs, warmup=args.warmup,
    )


if __name__ == "__main__":
    main()
