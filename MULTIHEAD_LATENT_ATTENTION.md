# Multi-Head Latent Attention (MLA)

## Overview

This implementation uses **multi-head latent attention** inspired by DeepSeek's architecture. The key innovation is that each attention head has its own latent compression space, allowing for more expressive and diverse representations.

## Architecture Changes

### Before (Single Latent Space)
- All attention heads shared a single latent compression of dimension `kv_lora_rank`
- One shared normalization layer for all heads
- Total latent dimension: `kv_lora_rank` (e.g., 128)

### After (Multi-Head Latent Space)
- Each attention head has its own latent compression space
- Per-head normalization layers (one for each head)
- Total latent dimension: `n_heads × kv_lora_rank` (e.g., 8 × 128 = 1024)

## Key Components

### 1. Input Projection (`wkv_a`)
```python
# Projects input to per-head latent spaces plus shared positional embeddings
self.wkv_a = Linear(dim, n_heads * kv_lora_rank + qk_rope_head_dim)
```

### 2. Per-Head Normalization
```python
# Each head gets its own normalization layer
self.kv_norm = nn.ModuleList([RMSNorm(kv_lora_rank) for _ in range(n_local_heads)])
```

### 3. Output Projection (`wkv_b`)
```python
# Projects from per-head latents to keys and values
self.wkv_b = ColumnParallelLinear(
    n_heads * kv_lora_rank,
    n_heads * (qk_nope_head_dim + v_head_dim)
)
```

## Benefits

1. **Increased Expressiveness**: Each head can learn its own compressed representation
2. **Better Specialization**: Different heads can focus on different aspects of the input
3. **Maintained Efficiency**: Still uses low-rank compression to reduce KV cache size
4. **Scalability**: Works with distributed training (tensor parallelism)

## Performance Characteristics

- **Parameter Count**: Increases by approximately `n_heads × kv_lora_rank` parameters for normalization layers
- **KV Cache**: Stores `n_heads × kv_lora_rank` values per token (per-head latent states)
- **Computation**: Slightly increased due to per-head normalization, but still efficient

## Configuration

The multi-head latent attention is controlled by the following parameters in `ModelArgs`:

- `n_heads`: Number of attention heads (default: 8)
- `kv_lora_rank`: Latent dimension per head (default: 128)
- `qk_nope_head_dim`: Non-positional key/query dimension (default: 64)
- `qk_rope_head_dim`: Rotary positional embedding dimension (default: 32)
- `v_head_dim`: Value dimension per head (default: 64)

## References

This implementation is inspired by DeepSeek's multi-head latent attention mechanism, which improves upon standard multi-head attention by using per-head latent compression.
