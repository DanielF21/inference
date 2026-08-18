"""Engine registry.

Imports live inside build_engine() so this module (and anything importing it)
stays free of torch and model loading — `--help` and analysis scripts must not
pull 3GB of weights onto a device.

Adding a new engine: one file here, one branch below.
"""

from __future__ import annotations

from infer.core import Engine, EngineConfig

ENGINE_NAMES = ("naive", "cached", "batched",)

# Engines that read cfg.pool_bytes. Anything else silently ignores it, which
# would record a run labelled as capped that never was.
POOL_AWARE = ("batched",)


def build_engine(name: str, cfg: EngineConfig) -> Engine:
    if cfg.pool_bytes is not None and name not in POOL_AWARE:
        raise ValueError(
            f"engine {name!r} does not honour a KV pool ceiling; only "
            f"{POOL_AWARE} does. Recording this run as capped would be false."
        )
    if name == "naive":
        from infer.engines.naive.engine import NaiveEngine

        return NaiveEngine(cfg)
    elif name == "cached":
        from infer.engines.cached.engine import CachedEngine

        return CachedEngine(cfg)
    elif name == "batched":
        from infer.engines.batched.engine import BatchedEngine

        return BatchedEngine(cfg)
    raise ValueError(f"unknown engine {name!r}; expected one of {ENGINE_NAMES}")
