# DeepSeek Multi-head Latent Attention (MLA)

## Overview

This implementation follows the **DeepSeek-V2/V3 Multi-head Latent Attention (MLA)** architecture as described in the original DeepSeek papers.

## Key Concept: Shared Latent Compression

The fundamental innovation of DeepSeek's MLA is **shared low-rank compression** of key-value (KV) pairs across ALL attention heads:

```
Traditional MHA: Each head stores full K, V → Large KV cache
DeepSeek MLA:    All heads share compressed latent → ~93% smaller KV cache
```

## Architecture Details

### Compression Phase (Forward Pass)
1. **Input projection to shared latent space:**
   ```python
   # Project to SHARED latent (not per-head!)
   wkv_a: dim → kv_lora_rank + qk_rope_head_dim
   # Example: 512 → 128 + 32 = 160
   ```

2. **Single normalization layer:**
   ```python
   kv_norm = RMSNorm(kv_lora_rank)  # One norm for all heads
   ```

3. **Expansion from shared latent to per-head K/V:**
   ```python
   # From shared latent to all heads' K and V
   wkv_b: kv_lora_rank → n_heads × (qk_nope_head_dim + v_head_dim)
   # Example: 128 → 8 × 128 = 1024
   ```

### KV Cache Structure
```python
# Stores SHARED latent vector, not per-head K/V!
kv_cache: (batch_size, seq_len, kv_lora_rank)
# NOT: (batch_size, seq_len, n_heads, kv_lora_rank)
```

## Why Shared Latent?

1. **Memory Efficiency:** Single latent vector per token instead of separate K/V for each head
2. **KV Cache Reduction:** Achieves ~93% reduction in KV cache size
3. **Preserved Expressiveness:** The shared latent is rich enough to reconstruct per-head K/V
4. **Better than MQA/GQA:** Maintains full modeling capacity unlike Multi-Query or Grouped-Query Attention

## Comparison

| Approach | KV Cache per Token | Shared? |
|----------|-------------------|---------|
| Standard MHA | `n_heads × head_dim × 2` | No |
| Multi-Query Attention | `1 × head_dim × 2` | Yes (K,V) |
| Grouped-Query Attention | `n_groups × head_dim × 2` | Partial |
| **DeepSeek MLA** | `kv_lora_rank` | **Yes (latent)** |

## Implementation in ByteGPT

```python
class MLA(nn.Module):
    def __init__(self, args):
        # Shared latent compression
        self.wkv_a = Linear(dim, kv_lora_rank + qk_rope_head_dim)
        
        # Single normalization for shared latent
        self.kv_norm = RMSNorm(kv_lora_rank)
        
        # Expand from shared latent to per-head K/V
        self.wkv_b = ColumnParallelLinear(
            kv_lora_rank, 
            n_heads * (qk_nope_head_dim + v_head_dim)
        )
```

## References

- DeepSeek-V2 Paper: [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
- DeepSeek-V3 Technical Report
- [Multi-head Latent Attention Explained](https://towardsdatascience.com/deepseek-v3-explained-1-multi-head-latent-attention-ed6bee2a67c4)
- [DeepSeek Wiki - Model Architecture](https://deepwiki.com/deepseek-ai/DeepSeek-V2/1.1-model-architecture)

## Important Note

**DO NOT** implement per-head latent compression. The whole point of DeepSeek's MLA is the **shared** latent space across all heads, which is what enables the dramatic KV cache reduction.
