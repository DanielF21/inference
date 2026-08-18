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
    p.add_argument("--batch-size", type=int, default=1,
                   help="sequences per forward pass; the prompt is replicated")
    p.add_argument("--pool-gib", type=float, default=None,
                   help="cap KV cache at N GiB per batch (default: device limit)")
    p.add_argument("--mixed-seed", type=int, default=None,
                   help="draw a ragged batch from this seed instead of "
                        "replicating one prompt; ignores --prompts")
    # Open-arrival mode: requests show up over time and queue, which is where
    # static batching's defect lives. See infer/arrivals.py.
    p.add_argument("--arrivals", action="store_true",
                   help="run an arrival-rate sweep instead of a benchmark")
    p.add_argument("--rates", default="1,3,5,7",
                   help="--arrivals: comma-separated arrivals per second")
    p.add_argument("--n-requests", type=int, default=400,
                   help="--arrivals: requests per rate")
    p.add_argument("--max-batch", type=int, default=32,
                   help="--arrivals: most sequences one batch may hold")
    p.add_argument("--runs", type=int, default=5, help="recorded runs per prompt")
    p.add_argument("--warmup", type=int, default=2, help="discarded-from-analysis runs")
    p.add_argument("--max-new-tokens", type=int, default=256)
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
    # results/ holds A10 runs only, so local dev-loop output must be able to
    # stay out of it. Without this the only way to smoke test is to record and
    # then delete, which is one forgotten step away from a contaminated file.
    p.add_argument("--no-write", action="store_true",
                   help="print results without touching results/")
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
    from infer.bench import run_benchmark, run_mixed_benchmark
    from infer.engines import build_engine
    from infer.runtime import device_name, resolve_device

    pool_bytes = int(args.pool_gib * 2**30) if args.pool_gib else None
    engine = build_engine(
        args.engine,
        EngineConfig(device=args.device, dtype=args.dtype, pool_bytes=pool_bytes),
    )

    if args.arrivals:
        from infer.arrivals import print_summary, run_sweep

        summaries, _ = run_sweep(
            engine,
            [float(r) for r in args.rates.split(",") if r.strip()],
            cfg=sampling,
            n_requests=args.n_requests,
            max_batch=args.max_batch,
            seed=args.seed,
            device_name=device_name(resolve_device(args.device)),
            write=not args.no_write,
        )
        print_summary(summaries)
        return

    common = dict(
        device=args.device, dtype=args.dtype, runs=args.runs, warmup=args.warmup,
        write=not args.no_write,
    )
    if args.mixed_seed is not None:
        from infer.workload import mixed_batch

        run_mixed_benchmark(
            engine, mixed_batch(args.batch_size, seed=args.mixed_seed,
                        max_new_tokens=args.max_new_tokens),
            sampling, **common,
        )
    else:
        run_benchmark(engine, prompts, sampling, batch_size=args.batch_size, **common)


if __name__ == "__main__":
    main()
