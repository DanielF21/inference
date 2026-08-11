"""FLOP and byte accounting per engine, with MFU and bandwidth utilization.

    uv run python scripts/roofline.py
    uv run python scripts/roofline.py --check

Wall-clock time is the only measured input; everything else is derived from
the architecture in results/naive/mfu_calculation.md. `--check` asserts that
the naive path still reproduces the MFU figures published there, so a change
to these formulas cannot silently rewrite history.

Two utilizations are reported because the two phases sit on opposite sides of
the roofline. MFU is meaningful for an engine dominated by recomputation;
once a cache removes that work MFU collapses by construction, and bandwidth
utilization is the number that carries information.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Architecture (Qwen2.5-1.5B). See results/naive/mfu_calculation.md section G1.
D_MODEL = 1536
N_LAYERS = 28
N_HEADS = 12
N_KV_HEADS = 2
HEAD_DIM = 128
FFN_DIM = 8960
VOCAB = 151_936

# Hardware (NVIDIA A10). Section G2.
PEAK_FLOPS = 125e12
BANDWIDTH = 600e9

BYTES_PER_PARAM = 2  # fp16


def _param_census() -> tuple[int, int]:
    """(body, embedding) parameter counts, derived rather than hardcoded."""
    kv_width = N_KV_HEADS * HEAD_DIM
    per_layer = (
        D_MODEL * D_MODEL          # q_proj
        + D_MODEL * kv_width       # k_proj
        + D_MODEL * kv_width       # v_proj
        + D_MODEL * D_MODEL        # o_proj
        + D_MODEL + 2 * kv_width   # q, k, v biases
        + 3 * D_MODEL * FFN_DIM    # gate, up, down
        + 2 * D_MODEL              # two RMSNorms
    )
    body = per_layer * N_LAYERS + D_MODEL  # final norm
    return body, VOCAB * D_MODEL


BODY_PARAMS, EMBED_PARAMS = _param_census()
TOTAL_PARAMS = BODY_PARAMS + EMBED_PARAMS
WEIGHT_BYTES = TOTAL_PARAMS * BYTES_PER_PARAM

# Attention has no parameters, so the L2 corollary does not apply. Per query
# head per layer, QK^T and A@V each cost 2·d_h·q·kv, giving 4·d·L·q·kv across
# all heads and layers. The causal half is skipped only when q == kv and the
# backend receives is_causal=True; with q == 1 every entry is required, so the
# halved coefficient applies to prefill alone.
ATTN_FULL = 4 * D_MODEL * N_LAYERS      # 172,032
ATTN_CAUSAL = ATTN_FULL // 2            # 86,016

# KV cache footprint per token: K and V, fp16, across every layer.
KV_BYTES_PER_TOKEN = 2 * BYTES_PER_PARAM * N_LAYERS * N_KV_HEADS * HEAD_DIM


def flops_naive(prompt: int, gen: int) -> dict[str, float]:
    """No cache: step i re-processes the whole sequence, n_i = prompt + i."""
    sum_n = sum(prompt + i for i in range(gen))
    sum_n2 = sum((prompt + i) ** 2 for i in range(gen))
    return {
        "body": 2 * BODY_PARAMS * sum_n,
        # logits_to_keep=1, so the vocab projection runs on one position per pass.
        "head": 2 * EMBED_PARAMS * gen,
        "attn": ATTN_CAUSAL * sum_n2,
    }


def flops_cached(prompt: int, gen: int) -> dict[str, float]:
    """Cached: one prefill over `prompt` tokens, then one token per step."""
    decode_steps = gen - 1
    # Prefill is q == kv == prompt and takes the causal saving; each decode
    # pass is a single query against the whole cache, which takes none.
    decode_kv = sum(prompt + j for j in range(1, decode_steps + 1))
    return {
        "body": 2 * BODY_PARAMS * (prompt + decode_steps),
        "head": 2 * EMBED_PARAMS * gen,
        "attn": ATTN_CAUSAL * prompt**2 + ATTN_FULL * decode_kv,
    }


def decode_bytes_per_step(prompt: int, gen: int, cached: bool) -> float:
    """Mean bytes crossing the bus per decode step.

    Weights are read once per pass regardless of engine. A cached engine also
    re-reads the whole cache each step; a naive engine holds no cache and
    recomputes K and V instead, so it moves weights only.
    """
    if not cached:
        return WEIGHT_BYTES
    # Step j attends over prompt + j positions, for j = 1..gen-1.
    mean_kv_tokens = prompt + gen / 2
    return WEIGHT_BYTES + mean_kv_tokens * KV_BYTES_PER_TOKEN


def load() -> dict[tuple[str, int], dict]:
    paths = sorted(RESULTS_DIR.glob("*/runs.csv"))
    if not paths:
        raise SystemExit(f"no results under {RESULTS_DIR}/*/runs.csv")

    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for path in paths:
        with path.open() as f:
            for row in csv.DictReader(f):
                if row["is_warmup"] == "True":
                    continue
                # Every formula here is batch 1: flops_naive and flops_cached
                # count one sequence, and decode_bytes_per_step has no batch
                # term. Batched rows would be scored against the wrong
                # denominator and silently understate utilization.
                if int(row.get("batch_size") or 1) != 1:
                    continue
                groups[(row["engine"], int(row["prompt_tokens"]))].append(row)

    out = {}
    for key, rows in groups.items():
        params = json.loads(rows[0]["engine_params"])
        out[key] = {
            "total_s": statistics.mean(float(r["total_s"]) for r in rows),
            "decode_s": statistics.mean(float(r["decode_s"]) for r in rows),
            "gen": int(rows[0]["completion_tokens"]),
            "cached": params.get("kv_cache") == "true",
            "n": len(rows),
        }
    return out


def report(data: dict[tuple[str, int], dict]) -> None:
    header = (
        f"{'engine':<8} {'P':>5} {'body':>9} {'attn':>9} {'total':>10} "
        f"{'t':>8} {'TFLOPS':>8} {'MFU':>7} {'GB/step':>8} {'GB/s':>7} {'MBU':>7}"
    )
    print(header)
    print("-" * len(header))

    for (engine, prompt), m in sorted(data.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        gen = m["gen"]
        f = (flops_cached if m["cached"] else flops_naive)(prompt, gen)
        total = sum(f.values())
        achieved = total / m["total_s"]

        per_step = decode_bytes_per_step(prompt, gen, m["cached"])
        step_s = m["decode_s"] / (gen - 1)
        bw = per_step / step_s

        print(
            f"{engine:<8} {prompt:>5} {f['body'] / 1e12:>8.1f}T {f['attn'] / 1e12:>8.1f}T "
            f"{total / 1e12:>9.1f}T {m['total_s']:>7.2f}s {achieved / 1e12:>8.2f} "
            f"{100 * achieved / PEAK_FLOPS:>6.1f}% {per_step / 1e9:>8.2f} "
            f"{bw / 1e9:>7.1f} {100 * bw / BANDWIDTH:>6.1f}%"
        )


# Published in results/naive/mfu_calculation.md section VI. If a formula change
# moves these, the analysis that cites them is no longer supported by the code.
NAIVE_PUBLISHED = {
    16: {"sum_n": 36_736, "sum_n2": 6_669_696, "total_tf": 97.0, "mfu": 13.8},
    64: {"sum_n": 49_024, "sum_n2": 10_786_176, "total_tf": 129.5, "mfu": 18.3},
    256: {"sum_n": 98_176, "sum_n2": 39_048_576, "total_tf": 260.8, "mfu": 28.6},
    1024: {"sum_n": 294_784, "sum_n2": 340_841_856, "total_tf": 802.0, "mfu": 34.3},
    4096: {"sum_n": 1_081_216, "sum_n2": 4_567_913_856, "total_tf": 3226.5, "mfu": 36.3},
}


def check(data: dict[tuple[str, int], dict]) -> int:
    failures = 0

    def expect(label: str, got: float, want: float, tol: float) -> None:
        nonlocal failures
        ok = abs(got - want) <= tol
        failures += not ok
        print(f"  [{'ok' if ok else 'FAIL'}] {label}: got {got:,.4g}, want {want:,.4g}")

    print("parameter census vs results/naive/mfu_calculation.md L3")
    expect("body params", BODY_PARAMS, 1_310_340_608, 0)
    expect("embedding params", EMBED_PARAMS, 233_373_696, 0)
    expect("total params", TOTAL_PARAMS, 1_543_714_304, 0)
    expect("attn coefficient (causal)", ATTN_CAUSAL, 86_016, 0)
    expect("KV bytes/token", KV_BYTES_PER_TOKEN, 28_672, 0)

    print("\nnaive closed forms vs section L7")
    for prompt, want in NAIVE_PUBLISHED.items():
        gen = 256
        expect(f"P={prompt} sum_n", sum(prompt + i for i in range(gen)), want["sum_n"], 0)
        expect(
            f"P={prompt} sum_n2",
            sum((prompt + i) ** 2 for i in range(gen)),
            want["sum_n2"],
            0,
        )

    print("\nnaive totals and MFU vs section VI")
    for prompt, want in NAIVE_PUBLISHED.items():
        m = data.get(("naive", prompt))
        if m is None:
            print(f"  [SKIP] P={prompt}: no recorded naive runs")
            continue
        total = sum(flops_naive(prompt, m["gen"]).values())
        expect(f"P={prompt} total TF", total / 1e12, want["total_tf"], 0.1)
        mfu = 100 * (total / m["total_s"]) / PEAK_FLOPS
        # 0.1pp: the published table rounds time to 2dp before dividing.
        expect(f"P={prompt} MFU %", mfu, want["mfu"], 0.1)

    print(f"\n{'PASS' if not failures else f'{failures} FAILURE(S)'}")
    return failures


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--check", action="store_true",
                   help="assert the naive path reproduces the published MFU table")
    args = p.parse_args()

    data = load()
    if args.check:
        raise SystemExit(1 if check(data) else 0)
    report(data)


if __name__ == "__main__":
    main()
