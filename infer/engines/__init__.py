"""Engine registry.

Imports live inside build_engine() so this module (and anything importing it)
stays free of torch and model loading — `--help` and analysis scripts must not
pull 3GB of weights onto a device.

Adding a new engine: one file here, one branch below.
"""

from __future__ import annotations

from infer.core import Engine, EngineConfig

ENGINE_NAMES = ("naive",)


def build_engine(name: str, cfg: EngineConfig) -> Engine:
    if name == "naive":
        from infer.engines.naive import NaiveEngine

        return NaiveEngine(cfg)
    raise ValueError(f"unknown engine {name!r}; expected one of {ENGINE_NAMES}")
