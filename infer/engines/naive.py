"""Autoregressive decode with no KV cache.

`use_cache=False` means every step recomputes attention over the whole
sequence from scratch, so per-step cost grows with sequence length.

`logits_to_keep=1` avoids materializing a [seq_len, 151936] logit tensor
every step; only the last row is ever read.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from infer.core import DecodeTrace, EngineConfig, GenerationResult, SamplingConfig
from infer.runtime import MemoryProbe, load_model, make_sync, resolve_device
from infer.sampling import sample_next_token


class NaiveEngine:
    name = "naive"

    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.sync = make_sync(self.device)
        self.model, self.tokenizer, self.load_s = load_model(cfg.device, cfg.dtype)
        self.memory = MemoryProbe(self.device)

    def describe(self) -> dict[str, str]:
        return {"kv_cache": "false", "logits_to_keep": "1"}

    def generate(
        self, prompts: Sequence[str], cfg: SamplingConfig
    ) -> list[GenerationResult]:
        return [self._generate_one(p, cfg) for p in prompts]

    @torch.no_grad()
    def _generate_one(self, prompt: str, cfg: SamplingConfig) -> GenerationResult:
        torch.manual_seed(cfg.seed)

        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        prompt_tokens = ids.shape[-1]

        trace = DecodeTrace(self.sync)
        self.memory.reset()
        stopped_reason = "max_new_tokens"

        trace.start()
        for _ in range(cfg.max_new_tokens):
            out = self.model(input_ids=ids, use_cache=False, logits_to_keep=1)
            next_id = sample_next_token(out.logits[:, -1, :], cfg)
            ids = torch.cat([ids, next_id[:, None]], dim=-1)
            trace.mark()

            if cfg.stop_on_eos and next_id.item() == self.tokenizer.eos_token_id:
                stopped_reason = "eos"
                break

        peak_mem_bytes, peak_mem_source = self.memory.read()

        # Work in token ids throughout and decode once at the end. Decoding
        # one id at a time and re-encoding the string corrupts byte-level BPE
        # tokens that carry partial UTF-8 sequences.
        new_ids = ids[0, prompt_tokens:].tolist()

        return GenerationResult(
            text=self.tokenizer.decode(new_ids),
            token_ids=new_ids,
            prompt_tokens=prompt_tokens,
            total_s=trace.total_s,
            ttft_s=trace.ttft_s,
            stopped_reason=stopped_reason,
            token_times_s=list(trace.offsets),
            peak_mem_bytes=peak_mem_bytes,
            peak_mem_source=peak_mem_source,
        )
