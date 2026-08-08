"""Benchmark runner: warmup, repeats, provenance capture, CSV + JSONL sinks.

Nothing pre-aggregated is ever written. One row per run, aggregation happens
in scripts/analyze.py — averages computed at write time can't be un-averaged,
and hide the variance that matters.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infer.core import Engine, GenerationResult, SamplingConfig

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

CSV_FIELDS = [
    # identity
    "bench_id", "run_index", "is_warmup", "ts_utc",
    # provenance
    "torch_version", "transformers_version",
    # engine
    "engine", "engine_params", "model", "dtype", "device", "device_name", "load_s",
    # workload
    "prompt_label", "prompt_tokens", "max_new_tokens", "batch_size",
    "temperature", "top_k", "top_p", "min_p", "seed", "stop_on_eos",
    # results
    "completion_tokens", "stopped_reason", "ttft_s", "total_s", "decode_s",
    "decode_tps", "e2e_tps", "itl_p50_ms", "itl_p95_ms",
    # memory
    "peak_mem_bytes", "peak_mem_source",
    # integrity
    "output_sha8", "notes",
]


def _percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile. Returns nan for empty input."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, round(q / 100 * len(ordered) + 0.5) - 1))
    return ordered[idx]


def _session_metadata(engine: Engine, cfg_device: str, cfg_dtype: str) -> dict[str, Any]:
    import torch
    import transformers

    from infer.core import MODEL_NAME
    from infer.runtime import device_name, resolve_device

    device = resolve_device(cfg_device)

    return {
        "bench_id": uuid.uuid4().hex[:8],
        "ts_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "engine": engine.name,
        "engine_params": json.dumps(engine.describe(), sort_keys=True),
        "model": MODEL_NAME,
        "dtype": cfg_dtype,
        "device": device.type,
        "device_name": device_name(device),
        "load_s": round(getattr(engine, "load_s", float("nan")), 4),
    }


def _to_row(
    meta: Mapping[str, Any],
    result: GenerationResult,
    *,
    prompt_label: str,
    run_index: int,
    is_warmup: bool,
    cfg: SamplingConfig,
    batch_size: int,
) -> dict[str, Any]:
    itl = result.itl_ms
    return {
        **meta,
        "run_index": run_index,
        "is_warmup": is_warmup,
        "prompt_label": prompt_label,
        "prompt_tokens": result.prompt_tokens,
        "max_new_tokens": cfg.max_new_tokens,
        "batch_size": batch_size,
        "temperature": cfg.temperature,
        "top_k": cfg.top_k,
        "top_p": cfg.top_p,
        "min_p": cfg.min_p,
        "seed": cfg.seed,
        "stop_on_eos": cfg.stop_on_eos,
        "completion_tokens": result.completion_tokens,
        "stopped_reason": result.stopped_reason,
        "ttft_s": round(result.ttft_s, 6),
        "total_s": round(result.total_s, 6),
        "decode_s": round(result.decode_s, 6),
        "decode_tps": round(result.decode_tps, 4),
        "e2e_tps": round(result.e2e_tps, 4),
        "itl_p50_ms": round(statistics.median(itl), 4) if itl else "",
        "itl_p95_ms": round(_percentile(itl, 95), 4) if itl else "",
        "peak_mem_bytes": result.peak_mem_bytes if result.peak_mem_bytes is not None else "",
        "peak_mem_source": result.peak_mem_source,
        # Surfaces silent output changes across engines without storing text.
        "output_sha8": hashlib.sha256(result.text.encode()).hexdigest()[:8],
        "notes": "",
    }


def write_results(rows: list[dict[str, Any]], raw: list[dict[str, Any]]) -> Path:
    """Append to results/<engine>/runs.{csv,jsonl}. Returns the csv path.

    One directory per engine, so a run can never append into another engine's
    file. Both sinks stay append-only.
    """
    engines = {row["engine"] for row in rows}
    if len(engines) != 1:
        raise ValueError(f"expected rows from exactly one engine, got {engines}")

    out_dir = RESULTS_DIR / engines.pop()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "runs.csv"

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    with (out_dir / "runs.jsonl").open("a") as f:
        for record in raw:
            f.write(json.dumps(record) + "\n")

    return csv_path


def run_benchmark(
    engine: Engine,
    prompts: Mapping[str, str],
    cfg: SamplingConfig,
    *,
    device: str,
    dtype: str,
    runs: int = 5,
    warmup: int = 2,
    batch_size: int = 1,
    write: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Benchmark one engine over a set of prompts.

    Returns (rows, raw). With write=False nothing touches disk, so a remote
    worker can hand the records back to the caller to write.

    Warmup runs are recorded with is_warmup=True rather than discarded, so
    the warmup effect can be shown from the data instead of asserted.

    Every repeat uses the same seed: the work should be identical run to run,
    so observed spread reflects system noise only.
    """
    meta = _session_metadata(engine, device, dtype)
    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []

    for label, prompt in prompts.items():
        print(f"[bench] prompt={label!r} warmup={warmup} runs={runs}")

        for i in range(warmup + runs):
            is_warmup = i < warmup
            result = engine.generate([prompt], cfg)[0]

            row = _to_row(
                meta, result,
                prompt_label=label,
                run_index=i - warmup,  # negative during warmup
                is_warmup=is_warmup,
                cfg=cfg,
                batch_size=batch_size,
            )
            rows.append(row)
            raw.append({**row, "token_ids": result.token_ids,
                        "token_times_s": result.token_times_s, "text": result.text})

            tag = "warmup" if is_warmup else f"run {i - warmup + 1}/{runs}"
            print(
                f"[bench]   {tag}: ttft={result.ttft_s:.3f}s "
                f"total={result.total_s:.3f}s decode_tps={result.decode_tps:.2f}"
            )

    if write:
        path = write_results(rows, raw)
        print(f"[bench] wrote {len(rows)} rows -> {path}")
    return rows, raw
