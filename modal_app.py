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
) -> tuple[list[dict], list[dict], dict]:
    import torch

    from infer.bench import run_benchmark
    from infer.core import EngineConfig, SamplingConfig
    from infer.engines import build_engine

    engine = build_engine(engine_name, EngineConfig(device="cuda", dtype="float16"))

    # Read after load, before generation: this is the weights footprint alone.
    props = torch.cuda.get_device_properties(0)
    device_info = {
        "name": props.name,
        "total_memory_bytes": props.total_memory,
        "weights_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
    }

    rows, raw = run_benchmark(
        engine,
        prompts,
        SamplingConfig(**sampling),
        device="cuda",
        dtype="float16",
        runs=runs,
        warmup=warmup,
        write=False,
    )
    return rows, raw, device_info


@app.local_entrypoint()
def main(
    engine: str = "naive",
    prompts: str = "",
    runs: int = 5,
    warmup: int = 2,
    max_new_tokens: int = 256,
    seed: int = 0,
) -> None:
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
