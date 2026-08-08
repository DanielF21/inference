"""Logit filters and next-token selection.

All filters operate on the last dim and accept [B, V]. Everything runs in
fp32: fp16 has a 10-bit mantissa, and a cumsum over 151,936 values saturates
it badly enough to corrupt the top-p mask.
"""

from __future__ import annotations

import torch
from torch import Tensor

from infer.core import SamplingConfig

NEG_INF = float("-inf")


def filter_top_k(logits: Tensor, k: int) -> Tensor:
    """Keep only the k highest-scoring tokens."""
    k = min(k, logits.shape[-1])
    kth_value = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth_value, NEG_INF)


def filter_top_p(logits: Tensor, p: float) -> Tensor:
    """Nucleus sampling: keep the smallest set whose mass is >= p."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Shift by one so the token that crosses p is itself kept.
    mask = cumulative - sorted_logits.softmax(dim=-1) > p
    sorted_logits = sorted_logits.masked_fill(mask, NEG_INF)

    return torch.full_like(logits, NEG_INF).scatter_(-1, sorted_indices, sorted_logits)


def filter_min_p(logits: Tensor, p: float) -> Tensor:
    """Keep tokens whose probability is >= p * max_probability.

    Scales the cutoff with model confidence: strict on peaked distributions,
    permissive on flat ones.
    """
    probs = logits.softmax(dim=-1)
    threshold = probs.amax(dim=-1, keepdim=True) * p
    return logits.masked_fill(probs < threshold, NEG_INF)


def sample_next_token(logits: Tensor, cfg: SamplingConfig) -> Tensor:
    """[B, V] logits -> [B] token ids.

    temperature=None is greedy argmax — the benchmark path, and what makes
    cross-engine output comparison meaningful.
    """
    logits = logits.float()

    if cfg.temperature is None:
        return logits.argmax(dim=-1)

    logits = logits / cfg.temperature
    if cfg.top_k is not None:
        logits = filter_top_k(logits, cfg.top_k)
    if cfg.top_p is not None:
        logits = filter_top_p(logits, cfg.top_p)
    if cfg.min_p is not None:
        logits = filter_min_p(logits, cfg.min_p)

    return torch.multinomial(logits.softmax(dim=-1), num_samples=1).squeeze(-1)
