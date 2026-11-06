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

```python
import torch
import thriftattention as ta

q = torch.randn(1, 32, 32768, 128, device="cuda", dtype=torch.float16)
k = torch.randn(1, 32, 32768, 128, device="cuda", dtype=torch.float16)
v = torch.randn(1, 32, 32768, 128, device="cuda", dtype=torch.float16)

out = ta.attention(q, k, v)
```

Q is shaped `[batch, query_heads, query_len, head_dim]`; K and V are shaped `[batch, kv_heads, kv_len, head_dim]`.

### Approximate exponentials

`AttentionConfig(exp_approx=True)` (or `TransformersAttentionConfig(exp_approx=True)`) makes the tiled NVFP4
kernels form the softmax weights as powers of two with integer arithmetic and sum them on the tensor core,
instead of calling the exponential unit. It is off by default. It applies to the dense and Thrift FP4 passes at
head_dim 64 and 128; MXFP4, head_dim 256, and the single-query (decode) kernels keep ordinary exponentials.
`benchmarks/compare_exp_approx.py` and `benchmarks/compare_thrift_exp_approx.py` report its speed and accuracy
against the exact path and FP16 SDPA.

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

```bash
python -m pip install 'torch>=2.8.0'
python -m pip install -e . --no-build-isolation
```

## Contributing

Please see `Contributing.md`

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
