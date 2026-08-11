"""Static batching: B sequences share one forward pass.

The batch is formed once and runs until its slowest member finishes. Nothing
joins mid flight and no finished row is reclaimed, which is what the engines
after this one are measured against.

Prompts of different lengths are left padded, so every row's generation
frontier sits in the same column. Uniform batches skip padding entirely and
stay on the code path the cached engine was measured on.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers.cache_utils import Cache

from infer.core import DecodeTrace, EngineConfig, GenerationResult, SamplingConfig
from infer.engines.batched.cache import BatchedCacheLayer
from infer.runtime import MemoryProbe, load_model, make_sync, resolve_device
from infer.sampling import sample_next_token


class BatchedEngine:
    name = "batched"

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
            "padding_side": "left",
            "logits_to_keep": "1",
            "attn": getattr(self.model.config, "_attn_implementation", "unset"),
        }

    def generate(
        self, prompts: Sequence[str], cfg: SamplingConfig
    ) -> list[GenerationResult]:
        """Every prompt rides through one batch, unlike the engines before this
        one, which looped. At the one prompt the benchmark harness passes today
        the two are the same call."""
        return self.generate_batch(prompts, cfg)

    @torch.no_grad()
    def generate_batch(
        self, prompts: Sequence[str], cfg: SamplingConfig
    ) -> list[GenerationResult]:
        torch.manual_seed(cfg.seed)

        encoded = [self.tokenizer(p, return_tensors="pt").input_ids[0] for p in prompts]
        # Real lengths per row. The padded width is not a prompt length and
        # must never be reported as one.
        prompt_tokens = [int(e.shape[0]) for e in encoded]
        batch_size = len(encoded)

        ids, mask = self._pack(encoded)

        # Everything the batch can possibly need, measured on the widest row.
        max_len = ids.shape[1] + cfg.max_new_tokens
        cache = Cache(layers=[
            BatchedCacheLayer(max_len=max_len)
            for _ in range(self.model.config.num_hidden_layers)
        ])

        # Per row rather than per batch: a sequence that emits EOS stops
        # collecting tokens, but its row keeps being computed. That waste is
        # the measurement, so it must not be optimized away.
        #
        # A plain list, not a device tensor. `if finished[i]` on a CUDA tensor
        # forces an implicit .item(), so a tensor would cost B device syncs per
        # step — measurement noise that grows with B, in an engine whose whole
        # claim is that step time does not.
        finished = [False] * batch_size
        generated: list[list[int]] = [[] for _ in range(batch_size)]
        stopped = ["max_new_tokens"] * batch_size

        trace = DecodeTrace(self.sync)
        self.memory.reset()
        trace.start()

        # Only a padded batch supplies these. See _pack for why a uniform batch
        # must not.
        extra: dict[str, torch.Tensor] = {}
        if mask is not None:
            extra["attention_mask"] = mask
            # transformers derives positions as arange(L) broadcast across the
            # batch (modeling_qwen2.py:372-375), which assumes every row starts
            # at column 0. Left padding breaks that: a row padded to width w
            # has its first real token at column w - len. Counting real tokens
            # instead gives each row positions from 0, and clamp keeps the pad
            # columns at 0 rather than -1.
            extra["position_ids"] = (mask.cumsum(-1) - 1).clamp(min=0)

        # PREFILL
        out = self.model(
            input_ids=ids, use_cache=True, past_key_values=cache,
            logits_to_keep=1, **extra,
        )
        next_id = sample_next_token(out.logits[:, -1, :], cfg)
        self._collect(next_id, finished, generated, stopped, cfg)
        trace.mark()

        # DECODE. Prefill already emitted one token, so decode owes n-1.
        for _ in range(cfg.max_new_tokens - 1):
            if all(finished):
                break

            if mask is not None:
                # The new token is real for every row, including rows that have
                # already finished: their slots keep being computed.
                ones = torch.ones(batch_size, 1, dtype=mask.dtype, device=mask.device)
                mask = torch.cat([mask, ones], dim=-1)
                extra["attention_mask"] = mask
                # One query per row, so each row's position is its own count of
                # real tokens so far, minus one.
                extra["position_ids"] = mask.sum(-1, keepdim=True) - 1

            out = self.model(
                input_ids=next_id[:, None], use_cache=True,
                past_key_values=cache, logits_to_keep=1, **extra,
            )
            next_id = sample_next_token(out.logits[:, -1, :], cfg)
            self._collect(next_id, finished, generated, stopped, cfg)
            trace.mark()

        peak_mem_bytes, peak_mem_source = self.memory.read()

        # Timings describe the batch, not a row: every sequence in it saw the
        # same prefill and the same step boundaries.
        return [
            GenerationResult(
                text=self.tokenizer.decode(generated[i]),
                token_ids=generated[i],
                prompt_tokens=prompt_tokens[i],
                total_s=trace.total_s,
                ttft_s=trace.ttft_s,
                stopped_reason=stopped[i],
                token_times_s=list(trace.offsets),
                peak_mem_bytes=peak_mem_bytes,
                peak_mem_source=peak_mem_source,
            )
            for i in range(batch_size)
        ]

    def _pack(self, encoded: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Stack rows into [B, width]. Returns (ids, attention_mask or None).

        Padding is on the left so every row's last real token lands in the same
        column. Decode reads logits with [:, -1, :] and appends one token per
        row, so a right padded row would sample its next token from padding.

        A uniform batch skips padding and returns no mask.
        An explicit mask disables SDPA's enable_gqa and its is_causal skip
        (integrations/sdpa_attention.py:38 and :120), changing
        both the kernel and the attention FLOP count. A batch with nothing to
        mask should not pay for one, and this is what keeps a uniform batch on
        the same code path the cached engine was measured on.
        """
        width = max(int(e.shape[0]) for e in encoded)
        if all(int(e.shape[0]) == width for e in encoded):
            return torch.stack(encoded).to(self.device), None

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        ids = torch.full((len(encoded), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(encoded), width), dtype=torch.long)
        for i, row in enumerate(encoded):
            ids[i, width - row.shape[0]:] = row
            mask[i, width - row.shape[0]:] = 1

        return ids.to(self.device), mask.to(self.device)

    def _collect(
        self,
        next_id: torch.Tensor,
        finished: list[bool],
        generated: list[list[int]],
        stopped: list[str],
        cfg: SamplingConfig,
    ) -> None:
        """Append this step's token to every row that is still running.

        EOS is appended before the row is marked finished, matching the cached
        engine, which concatenates then tests.
        """
        for i, token in enumerate(next_id.tolist()):
            if finished[i]:
                continue
            generated[i].append(token)
            if cfg.stop_on_eos and token == self.tokenizer.eos_token_id:
                finished[i] = True
                stopped[i] = "eos"
