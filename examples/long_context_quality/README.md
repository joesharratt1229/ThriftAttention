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
