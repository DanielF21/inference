"""Device/dtype resolution, model loading, and memory probing.

All backend-specific branching lives here, so engine code stays
device-agnostic.
"""

from __future__ import annotations

import platform
import subprocess
from collections.abc import Callable
from functools import lru_cache
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from infer.core import MODEL_NAME

_DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def resolve_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(spec: str) -> torch.dtype:
    try:
        return _DTYPES[spec.lower()]
    except KeyError:
        raise ValueError(f"unknown dtype {spec!r}; expected one of {sorted(_DTYPES)}") from None


def make_sync(device: torch.device) -> Callable[[], None]:
    """Return a function that blocks until queued device work is done.

    Without this, timers measure kernel-launch dispatch rather than compute.
    """
    if device.type == "cuda":
        return torch.cuda.synchronize
    if device.type == "mps":
        return torch.mps.synchronize
    return lambda: None


def device_name(device: torch.device) -> str:
    """Human-readable chip/GPU name, for the benchmark record."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
        return f"Apple Silicon ({platform.machine()})"
    return platform.processor() or platform.machine()


@lru_cache(maxsize=2)
def load_model(device_str: str, dtype_str: str):
    """Load the model onto a device. Cached, so repeat calls are free.

    Returns (model, tokenizer, load_s), where load_s is cold-start cost.
    """
    device = resolve_device(device_str)
    dtype = resolve_dtype(dtype_str)

    t0 = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=dtype, local_files_only=True
    )
    model = model.to(device)
    model.eval()
    make_sync(device)()
    load_s = perf_counter() - t0

    return model, tokenizer, load_s


class MemoryProbe:
    """Best-effort peak-memory measurement, honest about what it measured.

    CUDA gives a true peak. MPS has no peak API at all (only current/driver
    allocated), so we report a driver-allocated delta as a high-water proxy
    and record the source, so a proxy is never mistaken for a real peak.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._baseline = 0

    def reset(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        elif self.device.type == "mps":
            self._baseline = torch.mps.driver_allocated_memory()

    def read(self) -> tuple[int | None, str]:
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated(), "cuda.max_memory_allocated"
        if self.device.type == "mps":
            delta = torch.mps.driver_allocated_memory() - self._baseline
            return max(delta, 0), "mps.driver_allocated_memory_delta(proxy)"
        return None, "unavailable"
