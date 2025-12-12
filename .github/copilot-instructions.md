# Copilot Coding Agent Instructions

## Project Overview
ByteGPT is a small GPT implementation using byte-level encoding/decoding. This project demonstrates transformer architecture with advanced features including FP8 quantization and Mixture of Experts (MoE) layers.

## Repository Structure
- **Root Directory**: Contains main Python modules
  - `model.py`: Core transformer model implementation with MoE support
  - `train.py`: Training loop and model training utilities
  - `generate.py`: Text generation utilities
  - `byte_tokenizer.py`: Byte-level tokenization functions
  - `fp8_ops.py`: FP8 quantization operations
  - `fp8_cast_bf16.py`: FP8 to BF16 casting operations
  - `kernel.py`: Custom CUDA kernels
  - `convert.py`: Model conversion utilities
  - `old_model.py`: Legacy model implementation
- **configs/**: Model configuration files (JSON format)
- **data/**: Training data and saved models
- **__pycache__/**: Python cache files (auto-generated, not committed)

## Technology Stack
- **Primary Language**: Python
- **Framework**: PyTorch (with distributed training support)
- **Key Libraries**:
  - `torch` and `torch.nn` for model implementation
  - `torch.distributed` for distributed training
  - `safetensors` for model serialization
  - `torch.nn.functional` for functional operations

## Code Style and Conventions

### Python Standards
- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values (as seen in existing code)
- Use docstrings with Args and Returns sections for all functions
- Prefer `torch.bfloat16` for model computations (unless working with FP8 features)

### Naming Conventions
- **Variables and functions**: Use snake_case (e.g., `max_batch_size`, `get_batch`)
- **Classes**: Use PascalCase (e.g., `ModelArgs`, `Transformer`)
- **Constants**: Use UPPER_SNAKE_CASE for module-level constants
- **Private methods**: Prefix with underscore (e.g., `_internal_method`)

### Model Architecture Conventions
- Use dataclasses for configuration objects (see `ModelArgs` in `model.py`)
- Layer naming should be descriptive and follow the pattern: `{component}_{sublayer}` (e.g., `attn_norm`, `ffn_gate`)
- Transformer blocks should maintain consistent structure across layers

### Documentation Standards
- All public functions must have docstrings explaining purpose, parameters, and return values
- Use Google-style docstrings with Args and Returns sections
- Include type information in docstrings even when type hints are present
- Add inline comments for complex algorithms or non-obvious implementations

## Development Workflow

### Testing and Validation
- Test model changes with small configurations before running full training
- Validate tensor shapes and data types in new operations
- Check for gradient flow in new layer implementations
- For FP8 operations, ensure proper scaling and dequantization

### Code Changes
- When modifying model architecture, update corresponding config files in `configs/`
- Preserve backward compatibility when possible
- For new layer types, follow the existing pattern of separating forward pass logic
- Ensure distributed training compatibility (check `world_size` and `rank` usage)

### Performance Considerations
- Prefer in-place operations where safe (e.g., `div_()` instead of `div()`)
- Use `torch.compile` compatible code patterns
- Avoid Python loops for tensor operations; use vectorized PyTorch operations
- Consider memory efficiency for large models (use gradient checkpointing where appropriate)

## Specific Technical Guidelines

### Byte-Level Tokenization
- Vocabulary size is fixed at 256 (one per byte)
- Use `encode()` for text to byte conversion
- Use `decode()` for byte to text conversion with error handling (`errors='replace'`)
- Maintain UTF-8 encoding consistency

### FP8 Quantization
- FP8 operations are in `fp8_ops.py`
- Use `act_quant` for activation quantization
- Use `weight_dequant` for weight dequantization
- Use `fp8_gemm` for FP8 matrix multiplication
- The `gemm_impl` global variable controls BF16 vs FP8 mode

### Attention Implementation
- Support both "naive" and "absorb" attention implementations (controlled by `attn_impl`)
- Implement rotary positional embeddings (RoPE) with configurable dimensions
- Use LoRA-style decomposition for Q/KV projections when `q_lora_rank` or `kv_lora_rank` > 0

### Mixture of Experts (MoE)
- MoE layers alternate with dense layers
- Route scale, expert groups, and activation functions are configurable
- Support both softmax and sigmoid routing functions
- Implement proper expert load balancing

## Common Tasks

### Adding New Features
1. Update `ModelArgs` dataclass if adding new hyperparameters
2. Implement the feature in the appropriate module
3. Update configuration files in `configs/`
4. Add docstrings and type hints
5. Test with minimal config before full-scale training

### Modifying Existing Components
1. Maintain backward compatibility with existing checkpoints when possible
2. Update docstrings to reflect changes
3. Verify distributed training compatibility
4. Check both BF16 and FP8 code paths if modifying numerical operations

### Bug Fixes
1. Identify the root cause before making changes
2. Add comments explaining the fix if non-obvious
3. Test the fix with relevant model configurations
4. Ensure the fix doesn't break existing functionality

## Build and Test Commands
- **Training**: `python train.py [arguments]`
- **Generation**: `python generate.py [arguments]`
- **Model Conversion**: `python convert.py [arguments]`

## Important Notes
- Do not commit `__pycache__/` or `data/model.pt` (handled by `.gitignore`)
- Model checkpoints should use `.safetensors` format for safe serialization
- Always check tensor device placement when working with distributed training
- Be mindful of memory usage with large models and batch sizes
- The codebase uses PyTorch's bfloat16 as the primary dtype for mixed precision training

## External Resources
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Safetensors Format](https://github.com/huggingface/safetensors)
- [DeepSeek-V2 Paper](https://arxiv.org/abs/2405.04434) (for MoE architecture inspiration)

## Security Considerations
- Validate file paths for model loading and saving
- Use `errors='replace'` in decode operations to handle malformed UTF-8
- Be cautious with pickle-based serialization (prefer safetensors)
- Sanitize user inputs in command-line arguments
