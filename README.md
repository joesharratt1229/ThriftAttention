# ThriftAttention

**ThriftAttention: Selective Mixed Precision for Long-Context FP4 Attention**

[arxiv.org/abs/2605.23081](https://arxiv.org/pdf/2605.23081)

<figure>
  <img width="1414" height="987" alt="pareto_frontier_131072" src="https://github.com/user-attachments/assets/25599d12-a851-4147-8e1a-36201eff4b04" />

  <figcaption>
    <strong>Figure 1:</strong> Pareto frontier of NLL recovery vs inference efficiency at 131k context length.
  </figcaption>
</figure>


## Average long-context performance

| Method | Mean score | Avg. recovery |
|---|---:|---:|
| FP4 | 0.247 | 0.0 |
| FP16 | 0.469 | 100.0 |
| Top-k = 5% | 0.452 | 94.2 |
| Top-k = 10% | 0.458 | 97.6 |
| Top-k = 25% | 0.459 | 96.5 |

Average performance of Qwen3-8B, Ministral3-8B and Llama3-8B on Helmet, Ruler and LongBench-V1 long context benchmarks

## Usage

The high-level `thriftattention.attention` API currently uses the full SM120
selective mixed-precision backend. The SM80/SM86 build currently exposes its
dense INT8 and packed INT4 kernels through the experimental CUDA extension
bindings; see the support table below.

```python
import torch
import thriftattention as ta

q = torch.randn(1, 32, 32768, 128, device="cuda", dtype=torch.float16)
k = torch.randn(1, 32, 32768, 128, device="cuda", dtype=torch.float16)
v = torch.randn(1, 32, 32768, 128, device="cuda", dtype=torch.float16)

out = ta.attention(q, k, v)
```

Q is shaped `[batch, query_heads, query_len, head_dim]`; K and V are shaped `[batch, kv_heads, kv_len, head_dim]`.

## Integration with Transformers library
```python
import torch
from transformers import AutoModelForCausalLM
from thriftattention.integrations.transformers import register_transformers_attention
attn = register_transformers_attention()
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", attn_implementation=attn, torch_dtype=torch.float16)
```


## Installation

Prerequisites:

- Python >=3.10
- CUDA toolkit >=12.8
- PyTorch >=2.8.0 built with CUDA >=12.8
- Transformers >=4.52
- A local CUDA toolkit whose major/minor version matches `torch.version.cuda`

```bash
python -m pip install 'torch>=2.8.0'
```

The installer automatically selects the visible GPU architecture. For a
reproducible build, select it explicitly:

```bash
# Ampere: A100 (SM80) and SM86 GPUs such as RTX 30-series, A10, and A40.
THRIFTATTENTION_CUDA_ARCH=sm80 python -m pip install -e . --no-build-isolation

# Blackwell: SM120 GPUs.
THRIFTATTENTION_CUDA_ARCH=sm120 python -m pip install -e . --no-build-isolation
```

Without `THRIFTATTENTION_CUDA_ARCH`, installation selects `sm80` for a visible
SM80/SM86 GPU and `sm120` for a visible SM120 GPU. It defaults to `sm120` when
no CUDA device is visible. The SM80 build contains both `sm_80` and `sm_86`
code. Each installation contains one architecture family; clean the previous
extension before switching variants:

```bash
rm -rf build src/thriftattention/_C*.so
python -m pip install -e . --no-build-isolation
```

### CUDA support

| Build | GPU capabilities | Current kernel scope |
|---|---|---|
| `sm80` | SM80 and SM86 | Dense causal/noncausal INT8 and packed signed INT4 attention, head dimensions 64/128, FP16/BF16 output; experimental extension API |
| `sm120` | SM120 | NVFP4/MXFP4 attention, single-query decode, block selection, and selective mixed-precision ThriftAttention through the high-level API |

The SM80 work in this release does **not** implement selective mixed precision.
Here, mixed precision means selecting important KV blocks and evaluating those
blocks at higher precision while the remaining blocks use the lower-precision
path. SM80 currently evaluates the whole attention operation with one quantized
format. FP32 accumulation/softmax and FP16/BF16 output do not by themselves make
it the selective mixed-precision algorithm described in the paper.

### SM80 verification

On an SM80/SM86 machine, build the extension and run the quantized correctness
suite with:

```bash
THRIFTATTENTION_CUDA_ARCH=sm80 python -m pip install -e . --no-build-isolation
python -m pytest tests/test_cuda_sm80_quantized_correctness.py -q
```

The optional benchmark compares the SM80 kernels with FlashAttention 2 when
`flash-attn` is installed, otherwise it uses PyTorch's forced FlashAttention
backend:

```bash
# Limit a source build to Ampere kernels. A matching prebuilt wheel may install instead.
FLASH_ATTN_CUDA_ARCHS=80 MAX_JOBS=4 \
  python -m pip install flash-attn --no-build-isolation
python benchmarks/benchmark_sm80_quantized.py
```

## Contributing

Please see `CONTRIBUTING.md`.

## Citation
If you use ThriftAttention library in your research please cite as:
```
@misc{sharratt2026thriftattention,
  title         = {{ThriftAttention}: Selective Mixed Precision for Long-Context {FP4} Attention},
  author        = {Sharratt, Joe},
  year          = {2026},
  eprint        = {2605.23081},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2605.23081},
}
```
