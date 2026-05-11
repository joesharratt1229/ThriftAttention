# ThriftAttention

ThriftAttention is a CUDA/Python library for selective mixed-precision causal attention.

Current scope:

- SM120 NVIDIA kernels under `csrc/cuda/sm120/`.
- NVFP4 quantization with FP8 scales.
- Block-mean top-k selection kept separate from quantization.
- A Python API around the kept causal attention path.

The fused quantize-and-mean experiment is intentionally not part of this library API.

## Install

```bash
pip install -e .
```

The current CUDA extension targets SM120 (`TORCH_CUDA_ARCH_LIST=12.0a`).

## API Sketch

```python
import thriftattention as ta

out = ta.attention(q, k, v, top_k=4)
baseline = ta.fp4_attention(q, k, v)
selected = ta.select_blocks(q, k, top_k=4)
```

Inputs are `torch.float16` CUDA tensors shaped `[batch, heads, seq, head_dim]`.
Supported `head_dim` values are `64` and `128`.
