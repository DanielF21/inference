# inference

An exercise in understanding LLM inference by building it.

A series of engines, each adding one concept to the previous starting from a
naive decode loop with no KV cache. Every engine ships with a benchmark and a
written analysis: what improved, why, and what became the bottleneck next.
The chain continues until it stops being interesting.

## Model

**Qwen2.5-1.5B** (fp16), fixed across every engine. The engine is the only
independent variable.

| | |
|---|---|
| Parameters | 1.54 B |
| Layers | 28 |
| Hidden size | 1536 |
| Attention heads | 12 |
| KV heads (GQA) | 2 |
| Head dim | 128 |
| Vocab | 151,936 |

## Memory arithmetic

Formulas from [Transformer Inference Arithmetic](https://kipp.ly/p/transformer-inference-arithmetic)
(kipply, 2022).

**Weights** — parameters × 2 bytes at 16-bit precision:

```
1.54e9 × 2 = 3.09 GB   (2.88 GiB, matching model.safetensors on disk)
```

**KV cache per token** — `2 · 2 · n_layers · n_heads · d_head`, where the
first 2 is the K and V pair and the second is bytes per fp16 value. Qwen2.5
uses grouped-query attention, so the head term is `n_kv_heads = 2`, not the
12 attention heads:

```
2 × 2 × 28 × 2 × 128 = 28,672 bytes = 28 KiB / token
```

| | |
|---|---|
| Per token | 28 KiB |
| Per 1K tokens | 28 MiB |
| Per 2048-token sequence | 56 MiB |

GQA is doing heavy lifting here: with full multi-head attention the same
model would cost `2 × 2 × 28 × 12 × 128` = 168 KiB/token, **6× more**. Cheap
KV means memory capacity is not the first thing that binds which shapes
which optimizations are worth building and in what order.

## Hardware

**NVIDIA A10 on Modal.** Local machines are for the dev loop only; every
recorded number comes from the A10 so results are comparable across engines.

[Options Considered](modal_chip_analysis.md)

| | |
|---|---|
| Memory | 24 GB GDDR6 |
| Bandwidth | 600 GB/s |
| fp16 dense | 125 TFLOPS |
| Ridge point | 125e12 ÷ 600e9 = **208 FLOP/byte** |

Chosen over the L4, which has near-identical fp16 compute (121 TFLOPS) and
the same 24 GB but **half the memory bandwidth** (300 GB/s). Decode is
bandwidth-bound, so the A10 is roughly 2× faster at the thing being
optimized, and cheaper per token despite the higher hourly rate. The L4's
one advantage is fp8 support, which Ampere lacks.

## Memory budget

Measured on the A10, not estimated:

| | |
|---|---|
| Card total (nominal) | 24 GB |
| Visible to CUDA | 22.06 GiB |
| Model weights (fp16) | 2.88 GiB |
| Allocator reserved after load | 3.06 GiB |
| **Free for KV cache** | **~19 GiB** |

The 2.88 GiB weights figure matches the arithmetic above exactly
(1.54e9 × 2 bytes). Roughly 0.3 GiB of the card is not visible to CUDA at
all, and the allocator reserves ~0.2 GiB beyond the weights themselves.

19 GiB at 28 KiB/token is ~710,000 tokens — around 340 concurrent 2K
sequences. Memory would never bind, so the memory-management work would have
nothing to show.

**The KV pool is therefore capped at 1 GiB**, which admits ~37,400 tokens:

| Sequence length | Concurrent sequences at 1 GiB |
|---|---|
| 512 | 73 |
| 2048 | 18 |
| 4352 (4096 prompt + 256 gen) | 8 |

vLLM exposes the same knob as `gpu_memory_utilization`. Sizing the pool
deliberately and reporting behaviour at the ceiling is more informative than
running with 19 GB free and never reaching one.

## Expected constraints

Consequences that follow from the numbers above, to be confirmed or falsified
by measurement rather than assumed:

- **Decode is bandwidth-bound at low batch.** One token requires reading all
  3.09 GB of weights: `3.09 / 600` ≈ 5.1 ms, a ceiling of ~194 tok/s at
  batch 1 regardless of kernel quality.
- **Batch ~208 is where decode stops being memory-bound.** Decode arithmetic
  intensity is roughly the batch size, so it takes ~208 concurrent sequences
  to reach the ridge. The 1 GiB pool admits far fewer than that at long
  context, so this setup stays memory-bound by construction.
- **KV traffic overtakes weight traffic at scale.** At batch 32 with 4K
  context, KV reads are `32 × 4096 × 28 KiB` ≈ 3.76 GB per step, exceeding
  the 3.09 GB of weights — the bottleneck shifts from weights to cache.
- **Prefill is compute bound, decode bandwidth bound.** The two phases sit on opposite
  sides of the roofline, so they are reported separately (TTFT vs
  inter-token latency)

## Workload

Prompts are sliced to exact token counts (16 / 64 / 256 / 1024 / 4096) from a
single source text, so prompt length is a clean 4× geometric axis rather than
an approximation. Regenerate with `scripts/build_prompts.py`. Default
generation length is 256 tokens.

## Layout

```
infer/
  core.py        config, result types, Engine protocol, decode timing
  runtime.py     device/dtype resolution, model loading, memory probing
  sampling.py    temperature, top-k, top-p, min-p
  bench.py       warmup, repeats, CSV + JSONL sinks
  engines/       one file per engine
scripts/
  analyze.py         aggregation, cross-engine speedup, output parity
  build_prompts.py   regenerates prompts.py at exact token counts
modal_app.py         runs a benchmark on an A10, writes results locally
```

Results are partitioned by engine:

```
results/naive/runs.csv     one row per run, append-only
results/naive/runs.jsonl   same rows + token_ids, per-token times, text
```

One row per run, nothing pre-aggregated, so raw runs stay recoverable.
`analyze.py` globs every engine's directory and pools them, so cross-engine
comparison needs no extra step.

```
uv run modal run modal_app.py --engine naive     # recorded runs (A10)
uv run python -m infer --engine naive            # local dev loop only
uv run python scripts/analyze.py
```
