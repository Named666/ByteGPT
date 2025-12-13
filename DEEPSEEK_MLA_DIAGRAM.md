================================================================================
DeepSeek Multi-head Latent Attention (MLA) Architecture Diagram
================================================================================

Standard Multi-Head Attention (MHA) vs DeepSeek MLA
────────────────────────────────────────────────────────────────────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│                    STANDARD MULTI-HEAD ATTENTION (MHA)                      │
└─────────────────────────────────────────────────────────────────────────────┘

Input: h (d_model = 512)
    │
    ├──────────────┬──────────────┬──────────────┬─────────────┐
    ▼              ▼              ▼              ▼             ▼
 Head 1         Head 2         Head 3    ...  Head 8
    │              │              │              │             │
 ┌──┴──┐       ┌──┴──┐       ┌──┴──┐       ┌──┴──┐       ┌──┴──┐
 │ Q,K │       │ Q,K │       │ Q,K │       │ Q,K │   ... │ Q,K │
 │  V  │       │  V  │       │  V  │       │  V  │       │  V  │
 └─────┘       └─────┘       └─────┘       └─────┘       └─────┘
   96×2          96×2          96×2          96×2           96×2  ← Cache size
   
Total Cache per token: 8 heads × (96 + 64) = 1,280 values


┌─────────────────────────────────────────────────────────────────────────────┐
│                  DEEPSEEK MULTI-HEAD LATENT ATTENTION (MLA)                 │
└─────────────────────────────────────────────────────────────────────────────┘

Input: h (d_model = 512)
    │
    ▼
┌───────────────────┐
│  W_DKV (wkv_a)    │  ← Compression to SHARED latent
│   512 → 128       │
└─────────┬─────────┘
          │
          ▼
    c_KV (128) ──────────────────┐  ← SHARED compressed latent
          │                       │
          │ RMSNorm (kv_norm)    │  ← Single norm for ALL heads
          │                       │
          ▼                       │
┌────────────────────┐            │
│  W_UK, W_UV        │            │  ← Per-head expansion weights
│    (wkv_b)         │            │
│   128 → 1024       │            │
└─────────┬──────────┘            │
          │                       │
    ┌─────┴──────┬───────┬────────┼────────┐
    ▼            ▼       ▼        ▼        ▼
 Head 1       Head 2  Head 3  Head 8   CACHE (128) ← SHARED!
    │            │       │        │
 ┌──┴──┐     ┌──┴──┐ ┌──┴──┐  ┌──┴──┐
 │ K,V │     │ K,V │ │ K,V │  │ K,V │  ← Expanded from shared latent
 └─────┘     └─────┘ └─────┘  └─────┘

Total Cache per token: 128 values (90% reduction!)


Key Differences:
────────────────────────────────────────────────────────────────────────────────
1. MHA: Each head stores full K,V → Large cache
   MLA: All heads share compressed latent → Small cache

2. MHA: No compression → 1,280 values/token
   MLA: Low-rank compression → 128 values/token

3. MHA: Independent K,V per head
   MLA: Shared latent c_KV, per-head expansion


Implementation Flow (ByteGPT):
────────────────────────────────────────────────────────────────────────────────

1. INPUT PROJECTION (wkv_a)
   Input (512) → Shared Latent (128) + RoPE (32)
   
2. NORMALIZATION (kv_norm)
   RMSNorm on shared latent (128)
   
3. EXPANSION (wkv_b)
   Shared Latent (128) → All Heads' K,V (1024)
   Each head gets: K (64) + V (64) = 128 dim
   
4. CACHE STORAGE
   Store: Shared latent (128) ✓
   NOT: Per-head K,V (1024) ✗
   
5. ATTENTION COMPUTATION
   - Project query to latent space
   - Compute attention in latent space
   - Project result back to head space


Memory Comparison:
────────────────────────────────────────────────────────────────────────────────

Standard MHA:  ████████████████████████████████ 1,280 values/token
DeepSeek MLA:  ███ 128 values/token

Reduction: 90.0% (10× smaller cache!)


Mathematical Formulation:
────────────────────────────────────────────────────────────────────────────────

Standard MHA:
  K_i = W_K^i @ h        (per head i)
  V_i = W_V^i @ h        (per head i)
  Cache: {K_1, ..., K_n, V_1, ..., V_n}  (2n tensors)

DeepSeek MLA:
  c_KV = W_DKV @ h       (compress to shared latent)
  K_i = W_UK^i @ c_KV    (expand from shared latent)
  V_i = W_UV^i @ c_KV    (expand from shared latent)
  Cache: {c_KV}          (1 shared tensor)


Benefits:
────────────────────────────────────────────────────────────────────────────────
✓ 90% KV cache reduction
✓ Faster inference (less memory bandwidth)
✓ Longer context support (same memory)
✓ Maintains model expressiveness
✓ No accuracy loss compared to MHA

================================================================================
