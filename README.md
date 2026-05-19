# ThriftAttention

**ThriftAttention: Selective Mixed Precision for Long-Context FP4 Attention**

Paper: TODO add arxiv Link

<figure>
  <img
    width="1414"
    height="987"
    alt="Pareto frontier of ThriftAttention decode speedup and quality recovery at 131k context"
    src="https://github.com/user-attachments/assets/46ec240a-40bf-4f5a-80a7-1d5832136801"
  />
  <figcaption>
    <strong>Figure 1:</strong> ThriftAttention approaches FP4 decode latency while preserving near-FP16 quality. Pareto frontier of performance vs inference efficiency trade-off at 131k context length on Qwen3-8B.
  </figcaption>
</figure>
\begin{tabular}{lcc}
\toprule


## Average long-context performance

| Method | Mean score | Avg. recovery |
|---|---:|---:|
| FP4 | 0.247 | 0.0 |
| FP16 | 0.469 | 100.0 |
| Top-k = 5% | 0.452 | 94.2 |
| Top-k = 10% | 0.458 | 97.6 |
| Top-k = 25% | 0.459 | 96.5 |

Average performance of Qwen3-8B, Ministral3-8B and Llama3-8B on Helmet, Ruler and LongBench-V1 long context benchmarks

## API

```python
import torch
import thriftattention as ta

q = torch.randn(1, 32, 32768, 128, device="cuda", dtype=torch.float16)
k = torch.randn(1, 32, 32768, 128, device="cuda", dtype=torch.float16)
v = torch.randn(1, 32, 32768, 128, device="cuda", dtype=torch.float16)

out = ta.attention(q, k, v)
```

Q is shaped `[batch, query_heads, query_len, head_dim]`; K and V are shaped `[batch, kv_heads, kv_len, head_dim]`.


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

## Citation
TODO
