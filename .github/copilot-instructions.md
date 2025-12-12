# Copilot Coding Agent Instructions

## Project Overview
ByteGPT: Small GPT using byte-level tokenization (vocab=256), with FP8 quantization and Mixture of Experts (MoE).

## Key Files
- `model.py`: Transformer with MoE support
- `train.py`: Training loop
- `byte_tokenizer.py`: Byte-level encode/decode
- `fp8_ops.py`: FP8 quantization ops
- `configs/`: Model configurations (JSON)

## Code Conventions
- **Style**: PEP 8, snake_case for functions/vars, PascalCase for classes
- **Docs**: Google-style docstrings with Args/Returns, type hints required
- **Data types**: Use `torch.bfloat16` (default) or FP8 via `gemm_impl` global
- **Config**: Use dataclasses (`ModelArgs` pattern)

## Technical Guidelines
- **Tokenization**: Fixed vocab=256, use `encode()`/`decode()` with `errors='replace'`
- **FP8**: Use `act_quant`, `weight_dequant`, `fp8_gemm` from `fp8_ops.py`
- **Attention**: Support "naive"/"absorb" modes, RoPE, LoRA decomposition when rank > 0
- **MoE**: Alternating dense/MoE layers, configurable routing (softmax/sigmoid)
- **Performance**: Prefer in-place ops, vectorized operations, `torch.compile` compatible

## Development
- Update `configs/` when changing model architecture
- Maintain backward compatibility with checkpoints
- Verify distributed training (check `world_size`, `rank` usage)
- Use `.safetensors` format (not pickle)
- Test with small configs before full training

## Commands
- Train: `python train.py [args]`
- Generate: `python generate.py [args]`
