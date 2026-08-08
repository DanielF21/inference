PROMPTS = {
    "short": "The capital of France is",
    "medium_short": (
        "The history of the Roman Empire is long and complex, spanning many "
        "centuries of conquest, governance, and cultural change."
    ),
    "medium": (
        "Machine learning is a subfield of artificial intelligence that focuses "
        "on building systems capable of learning from data. Rather than being "
        "explicitly programmed with rules, these systems identify patterns and "
        "improve their performance over time as they are exposed to more examples."
    ),
    "long": (
        "The transformer architecture, introduced in the paper \"Attention Is "
        "All You Need,\" revolutionized natural language processing by replacing "
        "recurrent and convolutional layers with a mechanism called "
        "self-attention. This allows the model to weigh the importance of "
        "different words in a sequence relative to each other, regardless of "
        "their distance apart. Unlike RNNs, which process tokens sequentially, "
        "transformers process entire sequences in parallel, making them far "
        "more efficient to train on modern hardware. The architecture consists "
        "of an encoder and a decoder, each built from stacked layers of "
        "multi-head attention and feed-forward networks, with residual "
        "connections and layer normalization stabilizing training across depth."
    ),
    "very_long": (
        "Inference-time optimization for large language models has become a "
        "critical area of systems research as model sizes and deployment costs "
        "continue to grow. Naive autoregressive decoding recomputes the full "
        "forward pass at every generation step, which is wasteful because most "
        "of the computation from previous steps could be reused. The key-value "
        "cache addresses this by storing the key and value projections computed "
        "during earlier steps, so that each new token only requires computing "
        "attention against the cached history rather than reprocessing the "
        "entire sequence from scratch. This turns an operation that scales "
        "quadratically with sequence length into one that scales roughly "
        "linearly, at the cost of additional GPU memory to store the cache. As "
        "sequences grow longer or as more requests are batched together, the "
        "memory footprint of the KV cache becomes the dominant constraint on "
        "throughput, which motivated the development of PagedAttention, a "
        "memory management technique borrowed from operating systems virtual "
        "memory design. Instead of allocating one large contiguous block of "
        "memory per sequence, PagedAttention divides the cache into fixed-size "
        "blocks that can be allocated non-contiguously and shared between "
        "requests, dramatically reducing memory fragmentation and enabling much "
        "higher batch sizes. Combined with continuous batching, which allows "
        "new requests to join a running batch as soon as GPU slots free up "
        "rather than waiting for a fixed batch to fully complete, modern "
        "inference engines like vLLM and SGLang are able to achieve throughput "
        "many times higher than naive implementations on identical hardware."
    ),
}
