# Long-Context Mini Examples

Small practical scripts compare `fp16`, `fp4`, and `thrift` without large eval installs.

```bash
pip install -e ".[hf]"
pip install -r examples/long_context_quality/requirements.txt
```

## Forward / NLL

Runs forward pass of chosen model and records mean NLL across token positions between fp4, fp16 and ThriftAttention. Pass `--text-file` for a local corpus.

```bash
python examples/long_context_quality/run_nll_mini.py --lengths 65536 --methods fp16,fp4,thrift
```

## Ruler

Runs mini evaluation of fp4 vs fp16 vs ThriftAttention across ruler tasks.

```bash
python examples/long_context_quality/run_ruler_mini.py --lengths 65536 --methods fp16,fp4,thrift
```


## HELMET Mini

Runs mini evaluation of fp4 vs fp16 vs ThriftAttention across HELMET tasks.

```bash
python examples/long_context_quality/run_helmet_mini.py --lengths 65536 --tasks json_kv,kilt_popqa_3,narrativeqa --methods fp16,fp4,thrift
```
