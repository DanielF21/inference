"""Autoregressive decode that implements KV cache.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from infer.core import DecodeTrace, EngineConfig, GenerationResult, SamplingConfig
from infer.runtime import MemoryProbe, load_model, make_sync, resolve_device
from infer.sampling import sample_next_token
from infer.engines.cached.cache import SimpleCacheLayer
from transformers.cache_utils import Cache


class CachedEngine:
    name = "cached"

    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.sync = make_sync(self.device)
        self.model, self.tokenizer, self.load_s = load_model(cfg.device, cfg.dtype)
        self.memory = MemoryProbe(self.device)

    def describe(self) -> dict[str, str]:
        return {
            "kv_cache": "true",
            "cache_layout": "preallocated_contiguous",
            "logits_to_keep": "1",
            "attn": getattr(self.model.config, "_attn_implementation", "unset"),
        }

    def generate(
        self, prompts: Sequence[str], cfg: SamplingConfig
    ) -> list[GenerationResult]:
        return [self._generate_one(p, cfg) for p in prompts]
    
    @torch.no_grad()
    def _generate_one(self, prompt: str, cfg: SamplingConfig) -> GenerationResult:
        torch.manual_seed(cfg.seed)

        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        prompt_tokens = ids.shape[-1]

        # The prompt size + the most tokens it can generate
        max_len = prompt_tokens + cfg.max_new_tokens

        # Layer for each attention layer
        cache = Cache(layers=[SimpleCacheLayer(max_len=max_len) for _ in range(self.model.config.num_hidden_layers)])
 
        trace = DecodeTrace(self.sync)
        self.memory.reset()
        stopped_reason = "max_new_tokens"
        trace.start()

        # PREFILL
        out = self.model(input_ids=ids, use_cache=True, past_key_values=cache, logits_to_keep=1)
        next_id = sample_next_token(out.logits[:, -1, :], cfg)
        ids = torch.cat([ids, next_id[:, None]], dim=-1)
        trace.mark()

        if cfg.stop_on_eos and next_id.item() == self.tokenizer.eos_token_id:
            stopped_reason = "eos"
        # Prefill already emitted one token, so decode owes n-1 and none at
        # all if prefill itself produced EOS.
        remaining = 0 if stopped_reason == "eos" else cfg.max_new_tokens - 1

        # DECODE
        for _ in range(remaining):
            out = self.model(input_ids=next_id[:, None], use_cache=True, past_key_values=cache, logits_to_keep=1)
            next_id = sample_next_token(out.logits[:, -1, :], cfg)
            ids = torch.cat([ids, next_id[:, None]], dim=-1)
            trace.mark()

            if cfg.stop_on_eos and next_id.item() == self.tokenizer.eos_token_id:
                stopped_reason = "eos"
                break

        peak_mem_bytes, peak_mem_source = self.memory.read()
        
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