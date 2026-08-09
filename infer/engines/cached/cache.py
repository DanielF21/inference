from transformers.cache_utils import CacheLayerMixin
import torch


class SimpleCacheLayer(CacheLayerMixin):
    is_sliding = False
    is_compileable = False

    def __init__(self, max_len: int, **kwargs) -> None:
        # Sets keys/values to None and is_initialized to False, which update()
        # relies on to know whether it still has to allocate.
        super().__init__(**kwargs)
        self.max_len = max_len

        # For batch size 1 we can just use a single position
        self.pos = 0

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        # key_states arrives as [B, n_kv_heads, seq, head_dim]
        # Allocate space for prompt_len + max_tokens it can generate
        allocated = [key_states.shape[0], key_states.shape[1], self.max_len, key_states.shape[3]]
        self.keys = torch.zeros(size=allocated, dtype=key_states.dtype, device=key_states.device)
        self.values = torch.zeros(size=allocated, dtype=value_states.dtype, device=value_states.device)
        self.is_initialized = True

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        # Slice into seq dimension indexed by position
        self.keys[:, :, self.pos:self.pos + key_states.shape[2]] = key_states
        self.values[:, :, self.pos:self.pos + value_states.shape[2]] = value_states
        self.pos += key_states.shape[2]

        # Return the slice of the cache that is now populated
        return self.keys[:, :, :self.pos], self.values[:, :, :self.pos]
    
        
    def get_seq_length(self) -> int:
        return self.pos

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        # Mask size needs how wide and where it starts
        # In our case attention is append only so we always start at position 0
        # since we're not doing anything like sliding window attention
        kv_length = self.get_seq_length() + query_length
        kv_offset = 0
        
        return kv_length, kv_offset
    
    def get_max_length(self) -> int:
        return self.max_len