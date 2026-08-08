"""Measure the per-forward-pass CPU cost directly, instead of fitting it.

    uv run modal run scripts/measure_overhead.py

Prefill wall-clock is flat across p16/p64/p256 despite 16x more arithmetic,
which implies a fixed per-pass cost. Rather than fit that number from the
benchmark, this measures it: a pass over a single token is the smallest
possible amount of GPU work, so whatever it costs is the floor.

Three timings separate the pieces:

  pipelined   sync only after all reps    -> max(cpu, gpu), the real floor
  serialized  sync after every rep        -> what DecodeTrace.mark() does
  cpu_only    clock stopped before sync   -> host cost with the GPU excluded

pipelined vs serialized is the cost of the harness's per-token instrumentation.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import modal  # noqa: E402

from modal_app import GPU, image  # noqa: E402

app = modal.App("inference-overhead", image=image)


@app.function(gpu=GPU, timeout=900)
def measure(reps: int = 50, seq_lens: tuple[int, ...] = (1, 256, 4096)) -> dict:
    import torch
    from torch.profiler import ProfilerActivity, profile

    from infer.core import EngineConfig
    from infer.engines import build_engine

    engine = build_engine("naive", EngineConfig(device="cuda", dtype="float16"))
    model = engine.model

    out: dict = {
        "attn_implementation": getattr(
            model.config, "_attn_implementation", "unset"
        ),
        "device": torch.cuda.get_device_name(0),
        "reps": reps,
        "by_seq_len": {},
    }

    with torch.no_grad():
        for n in seq_lens:
            ids = torch.randint(0, 1000, (1, n), device="cuda")

            def one() -> None:
                model(input_ids=ids, use_cache=False, logits_to_keep=1)

            for _ in range(10):  # warm: kernel autotune, allocator, cache
                one()
            torch.cuda.synchronize()

            # Pipelined: the CPU may run ahead of the GPU across reps.
            torch.cuda.synchronize()
            t0 = perf_counter()
            for _ in range(reps):
                one()
            torch.cuda.synchronize()
            pipelined = (perf_counter() - t0) / reps

            # Serialized: a hard barrier after each pass, as the harness does.
            torch.cuda.synchronize()
            t0 = perf_counter()
            for _ in range(reps):
                one()
                torch.cuda.synchronize()
            serialized = (perf_counter() - t0) / reps

            # CPU only: launches are async, so returning from one() means the
            # work is queued, not done. Drain first so the queue never
            # backpressures and the CPU is never blocked by the GPU.
            cpu_times = []
            for _ in range(reps):
                torch.cuda.synchronize()
                t0 = perf_counter()
                one()
                cpu_times.append(perf_counter() - t0)
            torch.cuda.synchronize()
            cpu_only = statistics.median(cpu_times)

            # Launch count, to turn "us per pass" into "us per launch".
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                one()
                torch.cuda.synchronize()
            launches = 0
            for e in prof.key_averages():
                t = getattr(e, "self_device_time_total", None)
                if t is None:
                    t = getattr(e, "self_cuda_time_total", 0)
                if t > 0:
                    launches += e.count

            out["by_seq_len"][n] = {
                "pipelined_ms": pipelined * 1000,
                "serialized_ms": serialized * 1000,
                "cpu_only_ms": cpu_only * 1000,
                "launches": launches,
            }

    return out


@app.local_entrypoint()
def main(reps: int = 50) -> None:
    r = measure.remote(reps)

    print(f"\ndevice   : {r['device']}")
    print(f"attn     : {r['attn_implementation']}")
    print(f"reps     : {r['reps']}\n")

    head = (f"{'seq_len':>8}{'pipelined':>12}{'serialized':>12}{'cpu_only':>11}"
            f"{'launches':>10}{'us/launch':>11}")
    print(head)
    print("-" * len(head))
    for n, m in r["by_seq_len"].items():
        per_launch = m["cpu_only_ms"] * 1000 / m["launches"] if m["launches"] else float("nan")
        print(f"{n:>8}{m['pipelined_ms']:>11.2f}ms{m['serialized_ms']:>11.2f}ms"
              f"{m['cpu_only_ms']:>10.2f}ms{m['launches']:>10}{per_launch:>11.1f}")

    one = r["by_seq_len"].get(1)
    if one:
        print(f"\nfloor (seq_len=1, pipelined) : {one['pipelined_ms']:.2f} ms"
              f"   vs 5.1 ms bandwidth-bound")
        print(f"instrumentation cost         : "
              f"{one['serialized_ms'] - one['pipelined_ms']:+.2f} ms/pass")
