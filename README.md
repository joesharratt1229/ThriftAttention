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

Tiled NVFP4 attention (`method="fp4"`) always uses a separate P scale for each 16-entry microblock, with either setting of `exp_approx`. The `microblock_p` argument remains accepted for compatibility; setting it to `False` does not disable scaling.

## Profiling

```bash
python benchmarks/e2e_profiling.py
```

Runs FP16 SDPA, NVFP4, and Thrift at 5%, 10%, and 25% FP16 budgets for
4096, 8192, 16384, 32768, 65536, and 131072 tokens. Each run compares both
ordinary exp and exp-approx for NVFP4 and every Thrift budget, using the same
inputs and selected blocks, against one FP16 SDPA baseline. Reports packed attention
and end-to-end attention timings, speedups over FP16 SDPA, and cosine
similarity to FP16 SDPA. End-to-end timings include quantization and block
selection. Results are saved to a timestamped CSV in `benchmarks/results/`.
Use `--fractions`, positional sequence lengths, or `--output` to customize
the run. The table and CSV label each row with its `exp_mode`.

To compare ordinary and approximate exponentials at the same Thrift budget:

```bash
python benchmarks/compare_thrift_exp_approx.py 4096 8192 16384 32768 --fraction 0.05
```

Both paths share the same selected blocks. `--top-k` selects a fixed number
of FP16 blocks instead. Approximation applies to the FP4 pass; selected
FP16/BF16 blocks keep ordinary exponentials. The public tiled NVFP4 API also
accepts `AttentionConfig(method="thrift", exp_approx=True)`.

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
