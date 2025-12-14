# Copilot Coding Agent Instructions

## Project Overview
ByteGPT: Small GPT using byte-level tokenization (vocab=256), with FP8 quantization and Mixture of Experts (MoE). Implements a Transformer with Multi-Head Latent Attention (MLA), alternating dense/MoE layers, and support for distributed training.

## Key Files
- `model.py`: Core Transformer model with MLA, MoE, parallel layers, and FP8 support
- `train.py`: Distributed training loop with AdamW optimizer and safetensors checkpoints
- `generate.py`: Text generation with sampling, supports interactive and batch modes
- `byte_tokenizer.py`: Byte-level encode/decode (vocab=256, UTF-8 bytes)
- `fp8_ops.py`: FP8 quantization operations (act_quant, weight_dequant, fp8_gemm)
- `kernel.py`: Triton kernels for optimized FP8 operations
- `convert.py`: Convert Hugging Face checkpoints to ByteGPT format
- `configs/`: JSON configuration files (e.g., `config.json` with ModelArgs)
- `data/`: Training data files (e.g., `tiny_shakespear.txt`, `kjvdat.txt`)

## Code Conventions
- **Style**: PEP 8, snake_case for functions/vars, PascalCase for classes
- **Docs**: Google-style docstrings with Args/Returns, type hints required
- **Data types**: Use `torch.bfloat16` (default) or FP8 via `gemm_impl` global
- **Config**: Use dataclasses (`ModelArgs` pattern) loaded from JSON
- **Imports**: Standard library first, then third-party (torch, safetensors), then local
- **Error handling**: Use assertions for critical checks, `errors='replace'` in decoding
- **Performance**: Prefer in-place ops, vectorized operations, `torch.compile` compatible

## Technical Guidelines
- **Tokenization**: Fixed vocab=256, use `encode()`/`decode()` with `errors='replace'`
- **FP8**: Use `act_quant`, `weight_dequant`, `fp8_gemm` from `fp8_ops.py`; Triton kernels in `kernel.py` for acceleration
- **Attention**: MLA with "naive"/"absorb" modes, RoPE (with YARN for long sequences), LoRA decomposition when `q_lora_rank > 0`
- **MoE**: Alternating dense/MoE layers, configurable routing (softmax/sigmoid), experts per layer split across ranks
- **Distributed**: Use `world_size`, `rank` from torch.distributed; parallel embeddings/layers split by `world_size`
- **Checkpointing**: Use `.safetensors` format (not pickle); load/save with `safetensors.torch`
- **Initialization**: Xavier uniform for Linear, uniform(-0.666, 0.666) for Embedding

## Development
- Update `configs/` when changing model architecture
- Maintain backward compatibility with checkpoints
- Verify distributed training (check `world_size`, `rank` usage)
- Test with small configs before full training
- Use Triton for custom kernels when performance-critical

## Commands
- Train: `python train.py --ckpt-path data/checkpoints --config configs/config.json --data-file data/tiny_shakespear.txt --epochs 10 --batch-size 8 --learning-rate 0.01 --save-path model.safetensors`
- Generate: `python generate.py --ckpt-path data/checkpoints --config configs/config.json --interactive --max-new-tokens 200 --temperature 0.2`
- Convert HF: `python convert.py <hf_ckpt_path> <save_path> <n_experts> <mp>`

## Common Patterns
- **ModelArgs**: Dataclass for all hyperparameters, loaded from JSON
- **Global variables**: `world_size`, `rank`, `block_size`, `gemm_impl`, `attn_impl` set at module level
- **Parallel layers**: `ParallelEmbedding`, `ColumnParallelLinear`, `RowParallelLinear` for distributed training
- **Quantization**: Check `weight.element_size() > 1` for quantized weights; use `Linear.dtype` for FP8
- **MoE routing**: `Gate` with `topk` selection, `Expert` modules, `shared_experts` always active
- **Attention caching**: "naive" uses separate k/v caches; "absorb" uses absorbed kv_cache/pe_cache
- **Generation**: Use `@torch.inference_mode()` for inference, sample with temperature
