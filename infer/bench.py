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
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infer.core import Engine, GenerationResult, PoolExhausted, SamplingConfig

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
    # Rows from one batch share a wall clock. batch_id is what lets analysis
    # collapse them back to it; without it the seconds get counted B times.
    "batch_id", "seq_index",
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
    batch_id: str,
    seq_index: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    itl = result.itl_ms
    return {
        **meta,
        "run_index": run_index,
        "is_warmup": is_warmup,
        "prompt_label": prompt_label,
        "prompt_tokens": result.prompt_tokens,
        # Per row, not from cfg: a ragged batch gives each sequence its own
        # budget, and the batch reserves cache for the largest of them.
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "batch_id": batch_id,
        "seq_index": seq_index,
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
        print(f"[bench] prompt={label!r} batch={batch_size} warmup={warmup} runs={runs}")

        for i in range(warmup + runs):
            is_warmup = i < warmup
            # One batch per run. At batch_size=1 this is the single-prompt call
            # every engine before this one made, so their rows are unchanged.
            batch_id = uuid.uuid4().hex[:8]
            try:
                results = engine.generate([prompt] * batch_size, cfg)
            except PoolExhausted as exc:
                # A refusal is a fact about the configuration, not a
                # measurement of one, so nothing is recorded. Every repeat
                # would be refused identically.
                print(f"[bench]   refused: {exc}")
                break

            for seq_index, result in enumerate(results):
                row = _to_row(
                    meta, result,
                    prompt_label=label,
                    run_index=i - warmup,  # negative during warmup
                    is_warmup=is_warmup,
                    cfg=cfg,
                    batch_size=batch_size,
                    batch_id=batch_id,
                    seq_index=seq_index,
                    max_new_tokens=cfg.max_new_tokens,
                )
                rows.append(row)
                raw.append({**row, "token_ids": result.token_ids,
                            "token_times_s": result.token_times_s, "text": result.text})

            # Aggregate over the batch, since every row shares one wall clock.
            # Identical to the per-result number when batch_size is 1.
            first = results[0]
            decode_tokens = sum(r.completion_tokens - 1 for r in results)
            decode_tps = decode_tokens / first.decode_s if first.decode_s > 0 else float("nan")

            tag = "warmup" if is_warmup else f"run {i - warmup + 1}/{runs}"
            print(
                f"[bench]   {tag}: ttft={first.ttft_s:.3f}s "
                f"total={first.total_s:.3f}s decode_tps={decode_tps:.2f}"
            )

    if write:
        path = write_results(rows, raw)
        print(f"[bench] wrote {len(rows)} rows -> {path}")
    return rows, raw


def run_mixed_benchmark(
    engine: Engine,
    batch: Sequence[tuple[str, Any]],
    cfg: SamplingConfig,
    *,
    device: str,
    dtype: str,
    runs: int = 5,
    warmup: int = 2,
    write: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Benchmark one ragged batch: mixed prompt lengths and mixed budgets.

    `batch` is (prompt_label, Request) pairs, normally from
    workload.mixed_batch. The same batch runs every repeat, so spread is system
    noise rather than a different draw.

    Separate from run_benchmark because the unit differs. There the unit is a
    prompt and the batch is one prompt replicated; here the batch is the unit
    and every row can carry a different label, prompt length, and budget.

    Requires an engine with generate_batch, which today means the batched one.
    """
    if not hasattr(engine, "generate_batch"):
        raise TypeError(f"engine {engine.name!r} has no generate_batch")

    meta = _session_metadata(engine, device, dtype)
    requests = [request for _, request in batch]
    labels = [label for label, _ in batch]
    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []

    lengths = sorted({r.max_new_tokens for r in requests})
    print(f"[bench] mixed batch={len(batch)} budgets={lengths} "
          f"warmup={warmup} runs={runs}")

    for i in range(warmup + runs):
        is_warmup = i < warmup
        batch_id = uuid.uuid4().hex[:8]
        try:
            results = engine.generate_batch(requests, cfg)
        except PoolExhausted as exc:
            print(f"[bench]   refused: {exc}")
            break

        for seq_index, (label, request, result) in enumerate(
            zip(labels, requests, results)
        ):
            row = _to_row(
                meta, result,
                prompt_label=label,
                run_index=i - warmup,
                is_warmup=is_warmup,
                cfg=cfg,
                batch_size=len(batch),
                batch_id=batch_id,
                seq_index=seq_index,
                max_new_tokens=request.max_new_tokens,
            )
            rows.append(row)
            raw.append({**row, "token_ids": result.token_ids,
                        "token_times_s": result.token_times_s, "text": result.text})

        first = results[0]
        decode_tokens = sum(r.completion_tokens - 1 for r in results)
        decode_tps = decode_tokens / first.decode_s if first.decode_s > 0 else float("nan")

        tag = "warmup" if is_warmup else f"run {i - warmup + 1}/{runs}"
        print(
            f"[bench]   {tag}: ttft={first.ttft_s:.3f}s "
            f"total={first.total_s:.3f}s decode_tps={decode_tps:.2f}"
        )

    if write:
        path = write_results(rows, raw)
        print(f"[bench] wrote {len(rows)} rows -> {path}")
    return rows, raw
