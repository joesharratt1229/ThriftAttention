# Long-Context Mini Examples

Small practical scripts compare `fp16`, `fp4`, and `thrift` without large eval installs.

```bash
pip install flash-attn --no-build-isolation
pip install -r examples/long_context_quality/requirements.txt
```

## Forward / NLL

Runs forward pass of chosen model and records mean NLL across token positions between fp4, fp16 and ThriftAttention. By default it streams real text from `emozilla/pg19`; pass `--text-file` for a local corpus.

```bash
python examples/long_context_quality/run_nll_mini.py --lengths 65536 --methods fp16,fp4,thrift
```

### Per-token NLL at long context

`run_nll_per_token.py` records the NLL of every token rather than a mean, and supports
tensor parallelism for lengths that do not fit on one GPU. Launch it under `torchrun`
and it shards attention heads and the MLP intermediate dimension across the mesh:

```bash
torchrun --nproc_per_node=2 examples/long_context_quality/run_nll_per_token.py --length 131072
```

Without `torchrun` it runs single-GPU exactly as before. Sizing for a 36B model in
bf16: weights are 67 GiB and the MLP activation peak is 21.5 GiB at 131072 tokens, so
that length needs at least two 96 GiB cards. Both `num_attention_heads` and
`num_key_value_heads` must divide the number of GPUs.

Block selection is per-head, so each rank selects exactly the blocks it would have
selected on one GPU and the sparsity pattern under test is unchanged. What does change is
float summation order, in the `o_proj` and `down_proj` all-reduces. Measured on the same
tokens against a single-GPU run, hidden states agree to `2e-6` relative in fp32 but only
`2e-2` in bf16, since bf16 rounding compounds over the depth of the model — worth about
`5e-5` on a document's mean NLL and `0.03` on any individual token's.

Absolute per-token NLL is therefore comparable only within one `tp` size, which
`summary.md` records in its heading. Deltas against the fp16 baseline are unaffected,
because both sides of the subtraction run under the same sharding.

Note that the thrift kernels cap at 2048 KV blocks, which at `block_size=64` makes
131072 the longest supported context.

If `torchrun` hangs at the first collective with the GPUs pinned at 100%, NCCL is stuck
on peer-to-peer. Workstation cards without NVLink advertise P2P (`can_device_access_peer`
returns `True`) but deadlock on it, especially when `nvidia-smi topo -m` reports `SYS`
between the GPUs. Route collectives through host shared memory instead:

```bash
export NCCL_P2P_DISABLE=1
```

## Ruler

Runs mini evaluation of fp4 vs fp16 vs ThriftAttention across ruler tasks.

```bash
python examples/long_context_quality/run_ruler_mini.py --lengths 65536 --methods fp16,fp4,thrift
```


## HELMET Mini

Runs mini evaluation of fp4 vs fp16 vs ThriftAttention across HELMET tasks.

The mini runner does not need a full HELMET install at runtime. For `json_kv` and `kilt_popqa_3`, point `HELMET_DATA_DIR` or `--helmet-data-dir` at a HELMET `data/` directory containing only the requested length files. `narrativeqa` is streamed with Hugging Face `datasets`, so it does not need local HELMET JSONL files.

Install the HELMET data with either the official full-data path:

```bash
git clone https://github.com/princeton-nlp/HELMET.git /path/to/HELMET
cd /path/to/HELMET
bash scripts/download_data.sh
export HELMET_DATA_DIR=/path/to/HELMET/data
```

Or download the HELMET archive and extract only the files used by the default `65536` mini run:

```bash
mkdir -p /path/to/helmet-mini-data
hf download princeton-nlp/HELMET data.tar.gz --repo-type dataset --local-dir /tmp/helmet-data
tar -xzf /tmp/helmet-data/data.tar.gz -C /path/to/helmet-mini-data \
  data/json_kv/test_k900_dep6.jsonl \
  data/kilt/popqa_test_1000_k440_dep6.jsonl \
  data/kilt/popqa_test_1000_k3_dep6.jsonl
export HELMET_DATA_DIR=/path/to/helmet-mini-data/data
```

For other `--lengths`, use the matching filenames from `dataset_utils/helmet_gen.py`.

```bash
python examples/long_context_quality/run_helmet_mini.py --lengths 65536 --tasks json_kv,kilt_popqa_3,narrativeqa --methods fp16,fp4,thrift
```
