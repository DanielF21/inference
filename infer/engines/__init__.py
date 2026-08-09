"""Engine registry.

Imports live inside build_engine() so this module (and anything importing it)
stays free of torch and model loading — `--help` and analysis scripts must not
pull 3GB of weights onto a device.

Adding a new engine: one file here, one branch below.
"""

from __future__ import annotations

from infer.core import Engine, EngineConfig

ENGINE_NAMES = ("naive", "cached",)


def build_engine(name: str, cfg: EngineConfig) -> Engine:
    if name == "naive":
        from infer.engines.naive.engine import NaiveEngine

        return NaiveEngine(cfg)
    elif name == "cached":
        from infer.engines.cached.engine import CachedEngine

        return CachedEngine(cfg)
    raise ValueError(f"unknown engine {name!r}; expected one of {ENGINE_NAMES}")
