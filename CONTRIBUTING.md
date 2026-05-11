# Contributing

Keep the public surface small and named after behavior, not experiments.

## Layout

- `src/thriftattention/`: Python API and validation.
- `csrc/bindings.cpp`: PyTorch extension bindings.
- `csrc/cuda/sm120/nvfp4/`: current SM120 NVFP4 CUDA implementation.
- `csrc/include/thriftattention/sm120/`: SM120 helper intrinsics shared by kernels.

## Kernel Conventions

- Use names like `thrift_attention_causal_nvfp4` and `fp4_attention_causal_nvfp4`.
- Keep block selection separate from quantization.
- Do not add fused quantize-and-mean entry points.
- Prefer shared helpers in `csrc/include/thriftattention/sm120/` over duplicating inline PTX wrappers.
- Add a new architecture directory only when there is real architecture-specific code.

## Python API

Expose stable behavior through `src/thriftattention/functional.py`, `selection.py`, and `quantization.py`.
Avoid exporting experimental tuning switches as public API.
