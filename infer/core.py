"""Config, result types, the Engine protocol, and decode-loop timing.

Imports nothing heavy — importing this module must never load a model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Protocol, runtime_checkable

MODEL_NAME = "Qwen/Qwen2.5-1.5B"


@dataclass(frozen=True)
class SamplingConfig:
    """How to turn logits into tokens, and when to stop.

    Benchmark defaults are greedy (temperature=None) with EOS ignored, so
    every repeat does identical work and observed variance is system noise
    rather than sampling randomness.
    """

    max_new_tokens: int = 256
    temperature: float | None = None  # None => greedy argmax
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    seed: int = 0
    stop_on_eos: bool = False


@dataclass(frozen=True)
class Request:
    """One prompt and the budget it is allowed to spend.

    SamplingConfig.max_new_tokens is one number for a whole call, so it cannot
    express a batch whose rows stop at different steps. That unevenness is what
    makes a static batch keep computing rows that are already done, which is
    the cost this engine exists to measure.
    """

    prompt: str
    max_new_tokens: int


@dataclass(frozen=True)
class EngineConfig:
    """Where the engine runs."""

    device: str = "auto"  # auto | mps | cuda | cpu
    dtype: str = "float16"
    # Ceiling on KV cache bytes for one batch; None means only the device
    # limits it. A deployment running a large model is scarce on cache memory
    # because the weights take the card. This model is small enough that ~19
    # GiB stays free, so scarcity has to be imposed to be studied at all.
    pool_bytes: int | None = None


class PoolExhausted(RuntimeError):
    """A batch's KV cache would not fit the configured pool.

    Raised before anything is allocated, so a refusal costs nothing and says
    what it would have needed.
    """


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]
    prompt_tokens: int
    total_s: float
    ttft_s: float
    stopped_reason: str  # "max_new_tokens" | "eos"
    # Cumulative offsets from decode start, one per generated token.
    # Optional: an engine that doesn't expose token boundaries can't fill it.
    token_times_s: list[float] | None = None
    peak_mem_bytes: int | None = None
    peak_mem_source: str = "unavailable"

    @property
    def completion_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def decode_s(self) -> float:
        """Time after the first token — the sequential, memory-bound phase."""
        return self.total_s - self.ttft_s

    @property
    def decode_tps(self) -> float:
        """Throughput of the decode phase alone, excluding prefill.

        n-1 tokens, because the first token's cost is ttft, not decode.
        """
        if self.completion_tokens < 2 or self.decode_s <= 0:
            return float("nan")
        return (self.completion_tokens - 1) / self.decode_s

    @property
    def e2e_tps(self) -> float:
        if self.total_s <= 0:
            return float("nan")
        return self.completion_tokens / self.total_s

    @property
    def itl_ms(self) -> list[float]:
        """Inter-token latencies in ms. Empty when per-token timing is absent."""
        if not self.token_times_s or len(self.token_times_s) < 2:
            return []
        t = self.token_times_s
        return [(t[i] - t[i - 1]) * 1000.0 for i in range(1, len(t))]


@runtime_checkable
class Engine(Protocol):
    """One inference implementation.

    generate() takes and returns a sequence; an engine that handles one
    request at a time loops internally.
    """

    name: str

    def describe(self) -> dict[str, str]:
        """Engine-specific metadata (e.g. {"block_size": "16"}).

        Serialized into the engine_params CSV column, so an engine can record
        its own knobs without changing the schema.
        """
        ...

    def generate(
        self, prompts: Sequence[str], cfg: SamplingConfig
    ) -> list[GenerationResult]: ...


@dataclass
class DecodeTrace:
    """Times a decode loop from the inside.

    Only the engine knows where a token boundary is, so TTFT and inter-token
    latency can't be measured from outside the call. The device sync is
    injected rather than hardcoded so this works on any backend.
    """

    sync: Callable[[], None]
    offsets: list[float] = field(default_factory=list)
    _t0: float = 0.0

    def start(self) -> None:
        self.sync()
        self.offsets.clear()
        self._t0 = perf_counter()

    def mark(self) -> None:
        self.sync()
        self.offsets.append(perf_counter() - self._t0)

    @property
    def ttft_s(self) -> float:
        return self.offsets[0] if self.offsets else float("nan")

    @property
    def total_s(self) -> float:
        return self.offsets[-1] if self.offsets else float("nan")
