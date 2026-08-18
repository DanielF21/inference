"""Run a benchmark on a Modal A10; results are written to the local results/.

    uv run modal run modal_app.py --prompts p16 --runs 1 --warmup 1
    uv run modal run modal_app.py

The container returns records rather than writing them, so remote and local
runs land in the same results/runs.csv and stay directly comparable.
"""

from __future__ import annotations

import modal

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
HF_HOME = "/cache"
GPU = "A10"


def _download_weights() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_NAME)


# Versions are pinned to the local venv. They are recorded per row in the CSV,
# so letting them drift between engines would be a silent confounder.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.13.0",
        "transformers==5.14.1",
        "accelerate==1.14.0",
        "huggingface_hub",
    )
    # Set before the download so build and run resolve the same cache path;
    # load_model() uses local_files_only=True and will fail if they differ.
    .env({"HF_HOME": HF_HOME})
    .run_function(_download_weights)
    .add_local_python_source("infer")
)

app = modal.App("inference-bench", image=image)


@app.function(gpu=GPU, timeout=3600)
def bench_remote(
    engine_name: str,
    prompts: dict[str, str],
    sampling: dict,
    runs: int,
    warmup: int,
    batch_size: int = 1,
    pool_bytes: int | None = None,
    mixed_seed: int | None = None,
) -> tuple[list[dict], list[dict], dict]:
    import torch

    from infer.bench import run_benchmark, run_mixed_benchmark
    from infer.core import EngineConfig, SamplingConfig
    from infer.engines import build_engine
    from infer.workload import mixed_batch

    engine = build_engine(
        engine_name,
        EngineConfig(device="cuda", dtype="float16", pool_bytes=pool_bytes),
    )

    # Read after load, before generation: this is the weights footprint alone.
    props = torch.cuda.get_device_properties(0)
    device_info = {
        "name": props.name,
        "total_memory_bytes": props.total_memory,
        "weights_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
    }

    cfg = SamplingConfig(**sampling)
    common = dict(device="cuda", dtype="float16", runs=runs, warmup=warmup, write=False)

    if mixed_seed is None:
        rows, raw = run_benchmark(engine, prompts, cfg, batch_size=batch_size, **common)
    else:
        # Ragged on both axes. Built here rather than shipped over the wire so
        # only the seed crosses, and the batch stays reproducible from it.
        rows, raw = run_mixed_benchmark(
            engine, mixed_batch(batch_size, seed=mixed_seed,
                         max_new_tokens=cfg.max_new_tokens), cfg, **common
        )
    return rows, raw, device_info


@app.function(gpu=GPU, timeout=3600)
def arrivals_remote(
    engine_name: str,
    sampling: dict,
    rates: list[float],
    n_requests: int,
    max_batch: int,
    seed: int,
    pool_bytes: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Open-arrival sweep on the A10. Returns (summaries, csv rows).

    Records come back rather than being written here, matching bench_remote,
    so remote and local sweeps land in the same results/<engine>/arrivals.csv.
    """
    from infer.arrivals import run_sweep
    from infer.core import EngineConfig, SamplingConfig
    from infer.engines import build_engine
    from infer.runtime import device_name, resolve_device

    engine = build_engine(
        engine_name,
        EngineConfig(device="cuda", dtype="float16", pool_bytes=pool_bytes),
    )
    return run_sweep(
        engine, rates,
        cfg=SamplingConfig(**sampling),
        n_requests=n_requests,
        max_batch=max_batch,
        seed=seed,
        device_name=device_name(resolve_device("cuda")),
        write=False,
    )


@app.local_entrypoint()
def arrivals(
    engine: str = "batched",
    rates: str = "1,3,5,7",
    n_requests: int = 400,
    max_batch: int = 32,
    max_new_tokens: int = 256,
    seed: int = 0,
    pool_gib: float = 0.0,
) -> None:
    """uv run modal run modal_app.py::arrivals --rates 1,3,5,7"""
    from infer.arrivals import print_summary, write_arrivals

    summaries, rows = arrivals_remote.remote(
        engine,
        {"max_new_tokens": max_new_tokens, "seed": seed},
        [float(r) for r in rates.split(",") if r.strip()],
        n_requests,
        max_batch,
        seed,
        int(pool_gib * 2**30) if pool_gib else None,
    )
    print_summary(summaries)
    if rows:
        print(f"[modal] wrote {len(rows)} records -> {write_arrivals(engine, rows)}")


@app.local_entrypoint()
def main(
    engine: str = "naive",
    prompts: str = "",
    runs: int = 5,
    warmup: int = 2,
    max_new_tokens: int = 256,
    seed: int = 0,
    batch_size: int = 1,
    pool_gib: float = 0.0,
    mixed_seed: int = -1,
) -> None:
    """--mixed-seed >= 0 draws a ragged batch instead of replicating a prompt,
    in which case --prompts is ignored. --pool-gib 0 leaves the pool unbounded.
    """
    from infer.bench import write_results
    from infer.prompts import PROMPTS

    labels = [s.strip() for s in prompts.split(",") if s.strip()] or list(PROMPTS)
    unknown = [label for label in labels if label not in PROMPTS]
    if unknown:
        raise SystemExit(f"unknown prompt label(s): {unknown}; have {list(PROMPTS)}")

    rows, raw, info = bench_remote.remote(
        engine,
        {label: PROMPTS[label] for label in labels},
        {"max_new_tokens": max_new_tokens, "seed": seed},
        runs,
        warmup,
        batch_size,
        int(pool_gib * 2**30) if pool_gib else None,
        mixed_seed if mixed_seed >= 0 else None,
    )

    gib = 2**30
    print(
        f"[modal] {info['name']}  "
        f"total={info['total_memory_bytes'] / gib:.2f} GiB  "
        f"weights={info['weights_bytes'] / gib:.2f} GiB  "
        f"reserved={info['reserved_bytes'] / gib:.2f} GiB"
    )
    write_results(rows, raw)
    print(f"[modal] wrote {len(rows)} rows")
