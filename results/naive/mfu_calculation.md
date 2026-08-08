# Deriving MFU from first principles

A complete derivation of the FLOP counts and utilization figures reported for
each engine. Everything below follows from the model architecture, the
hardware specification, and one measured quantity: wall-clock time.

---

## I. Givens

### G1 — Model architecture (Qwen2.5-1.5B)

| symbol | meaning | value |
|---|---|---|
| `d` | hidden size (`d_model`) | 1,536 |
| `L` | transformer layers | 28 |
| `H` | query heads | 12 |
| `H_kv` | key/value heads (GQA) | 2 |
| `d_h` | head dimension | 128 |
| `F` | MLP intermediate size | 8,960 |
| `V` | vocabulary size | 151,936 |
| — | embedding and LM head | tied (one matrix) |
| — | attention projections | biased on Q, K, V; unbiased on O |
| — | MLP | gated: `gate`, `up`, `down` |
| — | normalization | RMSNorm, 2 per layer + 1 final |

Note `H · d_h = 12 × 128 = 1,536 = d`, and `H_kv · d_h = 2 × 128 = 256`.

### G2 — Hardware (NVIDIA A10)

| symbol | meaning | value |
|---|---|---|
| `Π` | peak fp16 dense throughput | 125 × 10¹² FLOP/s |
| `β` | memory bandwidth | 600 × 10⁹ B/s |
| — | device memory | 24 GB |

### G3 — Workload

| symbol | meaning | value |
|---|---|---|
| `P` | prompt length in tokens | 16, 64, 256, 1024, 4096 |
| `N` | tokens generated per run | 256 |
| — | precision | fp16 (2 bytes/value) |
| — | decoding | greedy, fixed seed |

---

## II. Definitions

**D1 — FLOP.** One floating-point operation: a single multiply or a single
add.

**D2 — Token-forward.** One token passed through the model one time. The
natural unit of work for an engine whose cost scales with tokens processed
rather than tokens emitted.

**D3 — Achieved throughput.** Total FLOPs performed during a run divided by
that run's wall-clock duration.

```
        F(P)
R  =  ────────
         t
```

**D4 — MFU** (Model FLOPs Utilization). Achieved throughput as a fraction of
hardware peak.

```
MFU  =  R / Π
```

**D5 — Sequence length at step i.** In an engine with no KV cache, step `i`
processes the whole sequence built so far:

```
n_i  =  P + i        for  i = 0, 1, …, N−1
```

---

## III. Lemmas

### L1 — Cost of a matrix multiply

Let `C = A @ B` with `A : [n × d]` and `B : [d × m]`.

`C` has `n · m` entries. Each is a dot product of two length-`d` vectors,
requiring `d` multiplies and `d − 1` adds, i.e. `2d − 1 ≈ 2d` FLOPs.

```
FLOPs(matmul)  =  2 · n · d · m
```

### L2 — Cost of a linear layer

A linear layer holds a weight matrix `[d_in × d_out]`, therefore
`d_in · d_out` parameters. Applying it to `n` tokens is the matmul
`[n × d_in] @ [d_in × d_out]`. By L1:

```
FLOPs(linear)  =  2 · n · d_in · d_out  =  2 · n · (parameters in the layer)
```

**Corollary.** Summed over every linear layer, pushing `n` tokens through a
network of `N_params` parameters costs `2 · n · N_params`.

### L3 — Parameter census

Counted directly from G1, one transformer layer contains:

| component | shape | parameters |
|---|---|---|
| `q_proj` weight | `d × d` | 2,359,296 |
| `k_proj` weight | `d × H_kv·d_h` | 393,216 |
| `v_proj` weight | `d × H_kv·d_h` | 393,216 |
| `o_proj` weight | `d × d` | 2,359,296 |
| Q, K, V biases | `d + 2·H_kv·d_h` | 2,048 |
| `gate_proj` | `d × F` | 13,762,560 |
| `up_proj` | `d × F` | 13,762,560 |
| `down_proj` | `F × d` | 13,762,560 |
| 2 × RMSNorm | `2 · d` | 3,072 |
| **per layer** | | **46,797,824** |

Note the effect of GQA: `k_proj` and `v_proj` map into `H_kv · d_h = 256`
dimensions rather than `d = 1536`, making them 6× smaller than under
multi-head attention.

Totalling across the model:

```
body       =  46,797,824 × 28  +  1,536   =  1,310,340,608
                                (final norm)

embedding  =  V × d  =  151,936 × 1,536   =    233,373,696

total      =                                 1,543,714,304
```

**Verification against disk.** At 2 bytes per fp16 parameter the tensor
payload is `1,543,714,304 × 2 = 3,087,428,608` bytes. The file
`model.safetensors` measures 3,087,467,144 bytes; the difference of 38,536
bytes is the safetensors JSON header, which precedes the tensor data.

Define for use below:

```
B  =  1,310,340,608     (body parameters)
E  =    233,373,696     (embedding = LM head)
```

### L4 — Cost of the body, per forward pass

By the corollary to L2, restricted to body parameters:

```
FLOPs_body(n)  =  2 · B · n
```

### L5 — Cost of the LM head, per forward pass

The engine passes `logits_to_keep = 1`, so the vocabulary projection is
applied to the final position only, not to all `n` positions:

```
FLOPs_head  =  2 · 1 · d · V  =  2 · E  =  466,747,392
```

This term is independent of `n`. Applied to every position instead it would
be `2 · E · n`, which at `n = 4096` is 1.9 TFLOP per pass rather than
0.0005 TFLOP.

### L6 — Cost of attention, per forward pass

Attention contains two matmuls with no associated parameters, so L2 does not
apply. Per layer, per query head:

```
scores = Q @ Kᵀ  :  [n × d_h] @ [d_h × n]   →   2 · n · d_h · n  =  2 d_h n²
out    = A @ V   :  [n × n]   @ [n × d_h]   →   2 · n · n · d_h  =  2 d_h n²
```

Over `H` query heads and `L` layers:

```
FLOPs_attn(n)  =  4 · d_h · H · L · n²  =  4 · d · L · n²  =  172,032 · n²
```

GQA does not reduce this term. There remain `H = 12` query heads producing
12 distinct score matrices; the `H_kv = 2` key/value heads are broadcast
across them. GQA reduces storage and memory traffic, not attention
arithmetic.

Define:

```
A  =  4 · d · L  =  172,032
```

### L7 — Summation across the generation loop

By D5, over `N = 256` steps:

```
Σn   =  Σ (P + i)   =  N·P  +  Σi   =  256P + 32,640

Σn²  =  Σ (P + i)²  =  N·P² + 2P·Σi + Σi²
                    =  256P² + 65,280P + 5,559,680
```

using the closed forms

```
Σi   =  (N−1)N/2        =  255 · 256 / 2        =    32,640
Σi²  =  (N−1)N(2N−1)/6  =  255 · 256 · 511 / 6  = 5,559,680
```

---

## IV. Theorem

Total FLOPs for one run at prompt length `P`, generating `N = 256` tokens
without a KV cache:

```
F(P)  =  2·B·Σn  +  2·E·N  +  A·Σn²
         ───────     ──────     ──────
          body        head       attention
```

with `B = 1,310,340,608`, `E = 233,373,696`, `A = 172,032`, and `Σn`, `Σn²`
as given by L7.

By D3 and D4, for a measured wall-clock time `t`:

```
MFU  =  F(P) / (t · Π)
```

---

## V. Computation

### At P = 4096

**Sums** (L7):
```
Σn   =  256 × 4096 + 32,640
     =  1,048,576 + 32,640
     =  1,081,216

Σn²  =  256 × 4096²  +  65,280 × 4096  +  5,559,680
     =  4,294,967,296  +  267,386,880  +  5,559,680
     =  4,567,913,856
```

**Terms** (Theorem):
```
body  =  2 × 1,310,340,608 × 1,081,216  =  2.8335e15  =  2833.5 TF
head  =  2 ×   233,373,696 × 256        =  1.1949e11  =     0.1 TF
attn  =        172,032 × 4,567,913,856  =  7.8580e14  =   785.8 TF
                                                        ──────────
                                             F(4096)  =  3619.5 TF
```

**Throughput** (D3), with `t = 71.15 s` (mean of 5 recorded runs):
```
R  =  3.6195e15 / 71.15  =  5.087e13  =  50.9 TFLOPS
```

**Utilization** (D4):
```
MFU  =  50.9 / 125  =  40.7%
```

### At P = 16

```
Σn   =  256 × 16 + 32,640                              =     36,736
Σn²  =  256 × 256 + 65,280 × 16 + 5,559,680            =  6,669,696

body =  2 × 1,310,340,608 × 36,736      =   96.3 TF
head =                                       0.1 TF
attn =  172,032 × 6,669,696             =    1.1 TF
                                          ─────────
                               F(16)    =   97.5 TF

R    =  9.75e13 / 5.61  =  1.74e13      =  17.4 TFLOPS
MFU  =  17.4 / 125                      =  13.9%
```

---

## VI. Results

| `P` | `Σn` | `Σn²` | body | head | attn | total | `t` | `R` | MFU |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 36,736 | 6,669,696 | 96.3 | 0.1 | 1.1 | 97.5 TF | 5.61 s | 17.4 | 13.9% |
| 64 | 49,024 | 10,786,176 | 128.5 | 0.1 | 1.9 | 130.5 TF | 5.66 s | 23.0 | 18.4% |
| 256 | 98,176 | 39,048,576 | 257.3 | 0.1 | 6.7 | 264.1 TF | 7.29 s | 36.2 | 29.0% |
| 1024 | 294,784 | 340,841,856 | 772.5 | 0.1 | 58.6 | 831.3 TF | 18.69 s | 44.5 | 35.6% |
| 4096 | 1,081,216 | 4,567,913,856 | 2833.5 | 0.1 | 785.8 | 3619.5 TF | 71.15 s | 50.9 | 40.7% |
