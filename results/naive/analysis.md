# naive: autoregressive decode with no KV cache

The baseline. Every step runs a full forward pass over the entire sequence so
far, recomputing all keys and values from scratch. `use_cache=False` and
`logits_to_keep=1` are both explicit, so the next engine changes exactly one
variable.

**Setup:** Qwen2.5-1.5B fp16, NVIDIA A10 (Modal), greedy, seed 0, 256 new
tokens, 5 recorded runs + 2 warmup per prompt.

## Results

| prompt | ptok | ttft | total | decode tok/s | itl p95 | per-step | peak mem |
|---|---|---|---|---|---|---|---|
| p16 | 16 | 0.022s | 5.606s | 45.66 | 23.7 ms | 21.9 ms | 2.90 GiB |
| p64 | 64 | 0.021s | 5.661s | 45.22 | 25.3 ms | 22.1 ms | 2.91 GiB |
| p256 | 256 | 0.022s | 7.291s | 35.08 | 32.7 ms | 28.5 ms | 2.92 GiB |
| p1024 | 1024 | 0.062s | 18.694s | 13.69 | 81.4 ms | 73.0 ms | 2.96 GiB |
| p4096 | 4096 | 0.255s | 71.146s | 3.60 | 284.5 ms | 277.9 ms | 3.16 GiB |

`total` is the mean wall clock per run. `per-step`
is `total / 256`.

Throughput falls **12.7×** from the shortest prompt to the longest.

Run to run variance seems negligible the five p4096 runs span 71.126–71.185 s,
a spread of 0.08%. Output hashes are identical across all runs of a given
prompt. This mostly confirms the fact that greedy decoding is determinsitic w/ fixed seed.

## Why p16 and p64 are indistinguishable

45.66 vs 45.22 tok/s is a 1% difference across a 4× change in prompt length.
This can be explain by the shape of the work function.

Step `i` processes `P + i` tokens, so a full generation costs

```
Σ(P + i) for i in 0..255  =  256·P + 32,640
```

The constant 32,640 comes from the generated tokens alone. 

$$
\sum_{i=0}^{255} i = \frac{(N - 1)N}{2} = \frac{255 \times 256}{2} = 32{,}640
$$
and dominates until
`256·P` becomes comparable to it — around P = 128. Below that, the prompt is
nearly irrelevant and the cost is essentially fixed. Above it, the curve
bends sharply. The same reasoning explains why `ttft` is flat at 16, 64 and
256 tokens: prefill at those lengths is fixed overhead bound, not
compute bound.

## Where the time actually goes

The interesting result is that the engine becomes **more** hardware-efficient
as it becomes slower.

FLOPs are counted as: body `2 × 1.31e9 × Σn` (non-embedding parameters), LM
head `2 × 0.233e9 × 256` (once per step, because `logits_to_keep=1`), and
attention `172,032 × Σn²` for `QK^T` and `attn @ V` across all 28 layers.

| prompt | total FLOPs | attention share | total time | achieved | MFU |
|---|---|---|---|---|---|
| p16 | 97.5 TF | 1.2% | 5.606s | 17.4 TFLOPS | 13.9% |
| p64 | 130.5 TF | 1.4% | 5.661s | 23.0 TFLOPS | 18.4% |
| p256 | 264.1 TF | 2.5% | 7.291s | 36.2 TFLOPS | 29.0% |
| p1024 | 831.3 TF | 7.1% | 18.694s | 44.5 TFLOPS | 35.6% |
| p4096 | 3619.5 TF | 21.7% | 71.146s | 50.9 TFLOPS | 40.7% |

`achieved = total FLOPs / total time`, and `MFU = achieved / 125 TFLOPS`.
Time is the only measured quantity in this table; the FLOP counts are derived
from the architecture (see the full calculation: [MFU Calculation](mfu_calculation.md)).

MFU rises monotonically from 13.9% to 40.7% against the A10's 125 TFLOPS
fp16 peak. At short prompts each forward pass is only 16 to (16 + 256) tokens wide so
the matmuls are small. At p4096 the matmuls are way larger.

The crossover is computable. A pass over `n` tokens has arithmetic intensity
`n` — the weights are read once regardless of how many tokens ride along, so
intensity is just tokens per weight-read. Against the A10's ridge point of
208 FLOP/byte (see [Hardware](../../README.md#hardware)), any pass narrower than 208 tokens is bandwidth-bound. The
sequence grows during a run, so each prompt crosses that line partway
through:

| prompt | `n` over the run | steps below ridge |
|---|---|---|
| p16 | 16 → 271 | 75% |
| p64 | 64 → 319 | 56% |
| p256 | 256 → 511 | 0% |

**p256 is the first prompt that is fully compute bound**, which is where
the MFU curve bends hardest (18.4% → 29.0%).

Attention's share of total FLOPs grows from 1.2% to 21.7% as context grows,
which is the quadratic term becoming visible. It is still not dominant at 4K
for a model this small. The MLP is still the bulk of the work but the trend is
the reason attention kernels matter at longer contexts.

## The size of the redundancy

At p4096 the engine performs **1,081,216 token forwards** to emit 256 tokens.
An engine that caches keys and values would need `4096 + 256 = 4,352`: one
prefill pass plus a single token pass per step.

That is a **248× difference in token-forwards, i.e. 99.6% of the work is
recomputation** of values that cannot change, because causal masking means no
earlier position ever attends forward and its keys and values are therefore
invariant.

## Memory

Peak allocated memory barely moves: 2.90 GiB at p16 to 3.16 GiB at p4096,
against 2.88 GiB of weights. Everything above the weights is transient
activation memory, and nothing persists between steps — which is precisely
the trade being made. The naive engine spends compute to avoid holding state.

Memory is measured with `torch.cuda.max_memory_allocated()`, a true peak
rather than a sampled proxy.

## Bottleneck, and what follows

The binding constraint is redundant computation, not bandwidth, not capacity,
not kernel quality. Nothing about memory is under pressure — 19 GiB of the
card sits unused.

Removing the recomputation is the next change, and it makes two falsifiable
predictions:

1. **Cached decode should be roughly flat across prompt length.** Each step
   reads the same 2.88 GiB of weights regardless of context, with only a
   small additional term for reading the cache. Where naive falls 12.7×
   across the prompt range, cached should barely move.
2. **The speedup is a curve, not a number** — small at p16, large at p4096.
   Reporting a single "Nx faster" figure for this change would be
   meaningless without stating the sequence length it was measured at.

The bandwidth ceiling for a cached step is `2.88 GiB ÷ 600 GB/s` ≈ 5.1 ms,
or roughly 194 tok/s, against the 277.9 ms per step measured here at p4096.
Landing well short of that ceiling while staying flat would indicate host and
kernel launch overhead rather than bandwidth, since a 1.5B model gives each
step only a few milliseconds of GPU work to hide that overhead behind.
