# Long-Context Quality Mini Examples

These examples show the main intuition behind ThriftAttention on language workloads: full FP4 attention can drift as context grows, while ThriftAttention keeps a small configurable fraction of FP16 blocks to recover quality.

The default model for the language examples is `Qwen/Qwen3-8B`.

## Install

```bash
pip install -e ".[hf,plots]"
pip install -r examples/long_context_quality/requirements.txt
```

## NLL-Over-Length Headline Demo

Quick mode scores prefill-only per-token negative log-likelihood on a few long PG-19 chunks:

```bash
python examples/long_context_quality/run_nll_mini.py --preset quick
```

`--dataset pg19` resolves to a parquet PG-19 mirror first, avoiding the retired script-based dataset loader path in recent `datasets` releases.

Useful overrides:

```bash
python examples/long_context_quality/run_nll_mini.py \
  --model Qwen/Qwen3-8B \
  --dataset pg19 \
  --lengths 8192,32768,65536 \
  --methods fp16,fp4,thrift \
  --fractions 0.05,0.10,0.25 \
  --num-docs 1
```

Offline smoke input:

```bash
python examples/long_context_quality/run_nll_mini.py --dataset synthetic --lengths 8192 --methods fp16
```

The script writes:

```text
results/long_context_quality/<timestamp>-nll-mini/
  metrics.jsonl
  summary.md
  per_token/*.npz
  environment.json
```

Example summary table format, with placeholder values:

| tokens | fp16 | fp4 | thrift_5pct | thrift_10pct | thrift_25pct |
| --- | --- | --- | --- | --- | --- |
| 8,192 | 1.234 | 1.260 (+0.026) | 1.238 (+0.004) | 1.236 (+0.002) | 1.235 (+0.001) |

These numbers are illustrative only. Actual values depend on GPU, model revision, Transformers version, dataset split, sequence lengths, and kernel availability.

## RULER Mini

`run_ruler_mini.py` runs tiny synthetic generation tasks inspired by RULER:

- `needle`: retrieve a hidden key
- `variable_tracking`: remember a variable value
- `common_words`: aggregate a short word multiset

```bash
python examples/long_context_quality/run_ruler_mini.py --preset quick
```

This is a smoke test, not an official RULER score. For full RULER reproduction, start from `benchmarks/paper_reproduction/ruler.yaml` and match the paper hardware/model/settings.

## HELMET Mini

`run_helmet_mini.py` generates a tiny HELMET-style config for representative tasks:

- `json_kv`
- `retrieval`
- `long_qa`

```bash
python examples/long_context_quality/run_helmet_mini.py --preset quick
```

It writes a generated config and `run_commands.sh` under `results/long_context_quality/`. Full HELMET evaluation is intentionally not the default.

## Quick vs. Standard vs. Paper-Scale

Quick presets are for first-run validation and should finish with a small number of documents/examples. Standard presets increase lengths or examples but are still not paper-scale. Paper-scale reproduction can take hours or days and should use the configs under `benchmarks/paper_reproduction/`.

## Hardware Notes

The fp16 baseline can run wherever the selected Transformers model runs. The `fp4` and `thrift` methods require the ThriftAttention CUDA extension and currently target SM120 consumer Blackwell GPUs. Unsupported hardware is reported as skipped rows instead of a cryptic kernel crash.
