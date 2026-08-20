#!/usr/bin/env python3
"""Per-token NLL: full-precision attention vs ThriftAttention mixed precision vs pure FP4.

One teacher-forced forward pass per method (use_cache=False, no KV cache), computing
the NLL of every token. Per-token arrays are saved to an .npz for offline analysis.

Methods:
  fp16        full-precision baseline (--baseline-impl, default flash_attention_2)
  local       thrift: local/diagonal blocks in fp16, remainder fp4
  quest       thrift: quest-selected blocks in fp16, remainder fp4
  block_mean  thrift: block-mean-selected blocks in fp16, remainder fp4
  *_drop      quest_drop/block_mean_drop/local_drop: selected blocks in fp16, the rest
              dropped entirely (PyTorch simulation). Runs at the compute-equivalent
              kept fraction f + (1-f)/--fp4-speedup so the FLOP budget matches the
              thrift run at fraction f.
  fp4         everything in fp4
  probe       fp16 forward pass (NLL should match the fp16 row) that additionally
              records, per layer/head/64x64 tile, the softmax quantisation error
              ||P_fp16 - P_fp4||_1, how much of it block_mean selection captures at
              each budget, the true mixed-precision residual, and cumulative
              error-concentration curves. See block_error_probe.py; analyse the
              saved probe/* arrays with analyze_block_error.py.

Budgets apply to the thrift/drop methods only. Smoke test on one RTX 6000 Pro:
  python run_nll_per_token.py --length 16384

Long contexts need tensor parallelism: a 36B model is 67 GiB of bf16 weights, and at
131072 tokens the MLP alone peaks at 21.5 GiB of activations, which does not fit
alongside them on one 96 GiB card. Launch under torchrun to shard attention heads and
the MLP intermediate dimension across GPUs -- halving both weights and activations, and
roughly halving wall-clock (attention is ~70% of the FLOPs at this length):
  torchrun --nproc_per_node=2 run_nll_per_token.py --length 131072

Block selection is per-head and per-query-block, so each rank picks exactly the blocks
it would have picked on one GPU: the sparsity pattern under test is unchanged. What does
change is float summation order, in the o_proj/down_proj all-reduces. Measured against a
single-GPU run of the same tokens: hidden states agree to 2e-6 relative in fp32, but only
2e-2 in bf16, because bf16 rounding compounds over the depth of the model. That lands as
~5e-5 on a document's mean NLL and ~0.03 on any individual token's.

So treat absolute per-token NLL as comparable only within one tp size -- summary.md
records it in the heading. Deltas against the fp16 baseline are unaffected, since both
sides of the subtraction run under the same sharding.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from statistics import fmean

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import block_error_probe
from thriftattention.integrations.transformers import (
    TransformersAttentionConfig,
    register_transformers_attention,
)
from thriftattention.selection import (
    select_block_pairs,
    select_local_block_pairs,
    select_quest_block_pairs,
)

BASELINE = "fp16"
BLOCK_SIZE = 64  # sm120 KV block size for head_dim 64/128
DROP_SELECT = {
    "quest": select_quest_block_pairs,
    "block_mean": select_block_pairs,
    "local": select_local_block_pairs,
}

RANK, LOCAL_RANK, WORLD_SIZE = (
    int(os.environ.get("RANK", 0)),
    int(os.environ.get("LOCAL_RANK", 0)),
    int(os.environ.get("WORLD_SIZE", 1)),
)


def log(*args: object, **kwargs: object) -> None:
    """Print from rank 0 only; every rank computes the same numbers."""
    if RANK == 0:
        print(*args, **kwargs, flush=True)


def setup_distributed() -> torch.device:
    device = torch.device("cuda", LOCAL_RANK)
    if WORLD_SIZE > 1:
        torch.cuda.set_device(device)
        if not dist.is_initialized():
            # device_id binds NCCL eagerly, so barrier() does not have to guess
            dist.init_process_group("nccl", device_id=device)
    return device


def check_tp_divisible(model_name: str) -> None:
    """Colwise q/k/v splits the head axis, so both head counts must divide the mesh."""
    if WORLD_SIZE == 1:
        return
    config = AutoConfig.from_pretrained(model_name)
    for attr in ("num_attention_heads", "num_key_value_heads"):
        heads = getattr(config, attr, None)
        if heads is not None and heads % WORLD_SIZE:
            raise SystemExit(f"{attr}={heads} is not divisible by world size {WORLD_SIZE}")


def broadcast_max(value: float, device: torch.device) -> float:
    if WORLD_SIZE == 1:
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-token NLL for ThriftAttention methods on long-context text.")
    parser.add_argument("--model", default="ByteDance-Seed/Seed-OSS-36B-Base")
    parser.add_argument("--dataset", default="emozilla/pg19")
    parser.add_argument("--length", type=int, default=16384)
    parser.add_argument(
        "--methods",
        default="fp16,fp4",
        help="comma list of fp16, local, quest, block_mean, quest_drop, block_mean_drop, local_drop, fp4, probe",
    )
    parser.add_argument("--budgets", default="0.05,0.10,0.25", help="fp16 block budgets for the thrift methods")
    parser.add_argument(
        "--fp4-speedup",
        type=float,
        default=4.0,
        help="assumed fp4:fp16 attention throughput ratio used to compute-match the *_drop budgets; inf runs them at --budgets as-is",
    )
    parser.add_argument("--num-docs", type=int, default=100)
    parser.add_argument("--ce-chunk", type=int, default=1024)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--baseline-impl", default="flash_attention_2", help="use sdpa if flash-attn is unavailable")
    parser.add_argument("--output", type=Path, default=Path("results/nll_per_token"))
    parser.add_argument("--seed", type=int, default=1234)
    probe = parser.add_argument_group("probe method")
    probe.add_argument("--probe-head-chunk", type=int, default=8, help="heads scored per score-matrix chunk")
    probe.add_argument("--probe-row-chunk", type=int, default=256, help="query rows per score-matrix chunk")
    probe.add_argument("--probe-layer-stride", type=int, default=1, help="measure every n-th layer")
    probe.add_argument("--probe-row-stride", type=int, default=1, help="measure every n-th row chunk")
    probe.add_argument(
        "--probe-save-tiles",
        action="store_true",
        help="save the full per-tile error map for the first document (large at long context)",
    )
    probe.add_argument("--probe-no-compile", action="store_true", help="disable torch.compile in the probe")
    probe.add_argument(
        "--probe-resid",
        action="store_true",
        help="also measure the mixed-precision residual (~1/3 slower; capture alone predicts recovery "
        "just as well while the capture-residual gap stays flat)",
    )
    return parser.parse_args()


def load_pg19_docs(tokenizer, *, dataset: str, length: int, num_docs: int, seed: int) -> list[list[int]]:
    from datasets import load_dataset

    required = length * num_docs
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.bos_token_id
    buffer: list[int] = []

    log(f"Loading {num_docs} x {length} tokens from {dataset}")
    ds = load_dataset(dataset, split="train", streaming=True).shuffle(seed=seed, buffer_size=200)
    for sample in ds:
        buffer.extend(tokenizer(sample["text"], add_special_tokens=False)["input_ids"])
        if eos is not None:
            buffer.append(eos)
        if len(buffer) >= required:
            break
    if len(buffer) < required:
        raise SystemExit(f"{dataset} only yielded {len(buffer)} tokens, but this run needs {required}.")
    return [buffer[i * length : (i + 1) * length] for i in range(num_docs)]


def load_docs(args: argparse.Namespace, device: torch.device) -> list[list[int]]:
    """Tokenise on rank 0 and broadcast, so every rank scores byte-identical documents."""
    docs = None
    if RANK == 0:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        docs = load_pg19_docs(
            tokenizer,
            dataset=args.dataset,
            length=args.length,
            num_docs=args.num_docs,
            seed=args.seed,
        )
    if WORLD_SIZE == 1:
        return docs

    tokens = torch.empty(args.num_docs, args.length, dtype=torch.int32, device=device)
    if RANK == 0:
        tokens.copy_(torch.tensor(docs, dtype=torch.int32))
    dist.broadcast(tokens, src=0)
    return tokens.cpu().tolist()


def build_runs(methods: list[str], budgets: list[float]) -> list[tuple[str, float | None]]:
    runs: list[tuple[str, float | None]] = []
    for method in methods:
        if method in ("fp16", "fp4", "probe"):
            runs.append((method, None))
        elif method in ("local", "quest", "block_mean") or (method.endswith("_drop") and method[:-5] in DROP_SELECT):
            runs.extend((method, budget) for budget in budgets)
        else:
            raise SystemExit(f"unknown method {method!r}")
    return runs


def run_label(method: str, budget: float | None) -> str:
    if budget is None:
        return method
    prefix = "eq" if method.endswith("_drop") else ""
    return f"{method}_{prefix}{budget * 100:g}pct"


def equivalent_fraction(budget: float, fp4_speedup: float) -> float:
    """Kept fraction whose fp16-only compute matches thrift at `budget`
    (fp16 on the budget + fp4 on the rest, fp4 being `fp4_speedup`x faster)."""
    return min(1.0, budget + (1.0 - budget) / fp4_speedup)


def register_drop_attention(selection: str, fraction: float) -> str:
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import flash_attention_mask

    name = f"drop_{selection}_{fraction:g}".replace(".", "p")
    if name not in AttentionInterface._global_mapping:
        AttentionInterface.register(name, partial(drop_attention_forward, selection=selection, fraction=fraction))
        AttentionMaskInterface.register(name, flash_attention_mask)
    return name


_flex_attention = torch.compile(flex_attention)
_DROP_KEEP: torch.Tensor | None = None  # module-level so the traced mask_mod stays one function


def _drop_mask_mod(b, h, q_idx, kv_idx):
    return _DROP_KEEP[h, q_idx // BLOCK_SIZE, kv_idx // BLOCK_SIZE] & (q_idx >= kv_idx)


def drop_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    *,
    selection: str,
    fraction: float,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Sparse fp16 attention via FlexAttention: selected KV blocks attended, the rest dropped."""
    if attention_mask is not None:
        raise RuntimeError("drop attention does not support explicit attention masks")
    batch, num_heads, seq_len, _ = query.shape
    if batch != 1:
        raise RuntimeError("drop attention simulation expects batch size 1")
    if seq_len % (2 * BLOCK_SIZE):
        raise RuntimeError(f"drop attention requires sequence length divisible by {2 * BLOCK_SIZE}")
    num_blocks = seq_len // BLOCK_SIZE

    selected = DROP_SELECT[selection](
        query,
        key,
        causal=True,
        fraction=fraction,
        block_size=BLOCK_SIZE,
        is_bf16=query.dtype == torch.bfloat16,
    ).long().view(num_heads, num_blocks, -1)
    # dense [head, q_block, kv_block] keep mask; -1 padding lands in the dummy last column
    keep = torch.zeros(num_heads, num_blocks, num_blocks + 1, dtype=torch.bool, device=query.device)
    keep.scatter_(2, selected.masked_fill(selected < 0, num_blocks), True)
    keep = keep[..., :-1].contiguous()

    # sm120's flex kernel only fits 64x64 tiles, which must divide the sparse block
    # size: build the BlockMask on 128-blocks (kept if either 64-half is kept) and let
    # mask_mod refine back down to the 64-token selection plus causality.
    global _DROP_KEEP
    _DROP_KEEP = keep
    half = num_blocks // 2
    keep128 = keep.view(num_heads, half, 2, half, 2).any(dim=4).any(dim=2)
    counts = keep128.sum(-1).to(torch.int32)
    order = keep128.byte().argsort(dim=-1, descending=True, stable=True).to(torch.int32)
    block_mask = BlockMask.from_kv_blocks(
        counts.unsqueeze(0),
        order.unsqueeze(0),
        BLOCK_SIZE=2 * BLOCK_SIZE,
        mask_mod=_drop_mask_mod,
        seq_lengths=(seq_len, seq_len),
    )
    out = _flex_attention(
        query,
        key,
        value,
        block_mask=block_mask,
        scale=scaling,
        enable_gqa=True,
        kernel_options={"BLOCK_M": 64, "BLOCK_N": 64},
    )
    return out.transpose(1, 2).contiguous(), None


def attention_impl(
    method: str,
    budget: float | None,
    args: argparse.Namespace,
    probe_state: block_error_probe.ProbeState | None = None,
) -> str:
    if method == BASELINE:
        return args.baseline_impl
    if method == "probe":
        return block_error_probe.register_probe_attention(probe_state)
    if method.endswith("_drop"):
        return register_drop_attention(method[:-5], equivalent_fraction(budget, args.fp4_speedup))
    return register_transformers_attention(
        TransformersAttentionConfig(
            name=f"thrift_{run_label(method, budget)}".replace(".", "p"),
            method="fp4" if method == "fp4" else "thrift",
            selection="block_mean" if method == "fp4" else method,
            fraction=budget or 0.0,
        )
    )


def load_model(args: argparse.Namespace):
    """Tensor-parallel under torchrun (SeedOssConfig ships a base_model_tp_plan), else one GPU."""
    kwargs = dict(dtype=getattr(torch, args.dtype), attn_implementation=args.baseline_impl)
    if WORLD_SIZE > 1:
        from transformers.distributed import DistributedConfig

        # distributed_config and device_map are mutually exclusive; TP places the shards.
        kwargs["distributed_config"] = DistributedConfig(tp_size=WORLD_SIZE)
    else:
        kwargs["device_map"] = "cuda"
    return AutoModelForCausalLM.from_pretrained(args.model, **kwargs).eval()


def summarize_probe(arrays: dict, budgets: list[float], num_docs: int, rows: list[dict]) -> list[dict]:
    """Pool the probe accumulators over docs, layers, and heads into per-budget fractions.

    `rows` supplies the NLL recovery each budget achieved, so the run prints the
    prediction (captured error) next to the outcome it should predict. The probe
    selects with block_mean, so it pairs with the block_mean rows; recovery is
    omitted unless those ran alongside fp16 and fp4.
    """
    def pooled(name: str, index=slice(None)) -> float:
        return float(sum(arrays[f"probe/{name}/doc{i}"][..., index].sum() for i in range(num_docs)))

    err_total = pooled("err_total")
    mass_total = pooled("mass_total")
    has_resid = pooled("err_resid") > 0
    nll = {row["label"]: row["mean_nll"] for row in rows}
    gap = nll["fp4"] - nll[BASELINE] if {"fp4", BASELINE} <= nll.keys() else 0.0
    summary = []
    for bi, budget in enumerate(budgets):
        entry = {
            "budget": budget,
            "err_capture": pooled("err_captured", bi) / err_total,
            "mass_capture": pooled("mass_captured", bi) / mass_total,
        }
        thrift = nll.get(run_label("block_mean", budget))
        if thrift is not None and gap:
            entry["nll_recovery"] = (nll["fp4"] - thrift) / gap
        if has_resid:
            entry["err_removed"] = 1.0 - pooled("err_resid", bi) / err_total
        summary.append(entry)
    return summary


@torch.no_grad()  # not inference_mode: torch.compile'd flex_attention rejects inference tensors
def per_token_nll(model, body, doc: list[int], ce_chunk: int, device: torch.device) -> torch.Tensor:
    """NLL of tokens 1..n-1 from a single forward pass, lm_head applied in chunks."""
    input_ids = torch.tensor([doc], dtype=torch.long, device=device)
    hidden = body(input_ids=input_ids, use_cache=False).last_hidden_state[:, :-1]
    targets = input_ids[0, 1:]
    lm_head = model.get_output_embeddings()
    return torch.cat(
        [
            F.cross_entropy(
                lm_head(hidden[:, start : start + ce_chunk]).squeeze(0).float(),
                targets[start : start + ce_chunk],
                reduction="none",
            )
            for start in range(0, targets.numel(), ce_chunk)
        ]
    )


def main() -> None:
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    budgets = [float(b) for b in args.budgets.split(",") if b.strip()]
    if args.length % BLOCK_SIZE:
        raise SystemExit(f"--length must be a multiple of {BLOCK_SIZE}")
    runs = build_runs(methods, budgets)
    if any(m.endswith("_drop") for m in methods):
        kept = ", ".join(f"{equivalent_fraction(b, args.fp4_speedup):g}" for b in budgets)
        log(f"drop methods keep fractions [{kept}] (compute-equivalent to {budgets} at {args.fp4_speedup:g}x fp4 speedup)")
    torch.manual_seed(args.seed)

    device = setup_distributed()
    check_tp_divisible(args.model)  # before the dataset download, so a bad mesh fails fast
    docs = load_docs(args, device)

    log(f"Loading {args.model} in {args.dtype}" + (f" (tensor parallel over {WORLD_SIZE} GPUs)" if WORLD_SIZE > 1 else ""))
    model = load_model(args)
    body = getattr(model, model.base_model_prefix)

    probe_state = None
    if any(method == "probe" for method, _ in runs):
        probe_state = block_error_probe.ProbeState(
            block_error_probe.ProbeConfig(
                budgets=tuple(budgets),
                baseline_impl=args.baseline_impl,
                num_layers=model.config.num_hidden_layers,
                head_chunk=args.probe_head_chunk,
                row_chunk=args.probe_row_chunk,
                layer_stride=args.probe_layer_stride,
                row_stride=args.probe_row_stride,
                compile=not args.probe_no_compile,
                resid=args.probe_resid,
            )
        )

    out_dir = None
    if RANK == 0:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out_dir = args.output / f"{stamp}-{args.model.rsplit('/', 1)[-1]}-{args.length}"
        out_dir.mkdir(parents=True)
        environment = vars(args) | {"tp_size": WORLD_SIZE}
        (out_dir / "environment.json").write_text(json.dumps(environment, indent=2, default=str) + "\n")

    arrays = {f"tokens/doc{i}": np.asarray(doc, dtype=np.int32) for i, doc in enumerate(docs)}
    rows = []
    for method, budget in runs:
        label = run_label(method, budget)
        model.set_attn_implementation(attention_impl(method, budget, args, probe_state))
        torch.cuda.reset_peak_memory_stats()
        seconds = []
        for i, doc in enumerate(docs):
            if method == "probe":
                probe_state.config.save_tiles = args.probe_save_tiles and i == 0
            if WORLD_SIZE > 1:
                dist.barrier()  # exclude rank skew from the per-doc timing
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            nll = per_token_nll(model, body, doc, args.ce_chunk, device)
            torch.cuda.synchronize()
            seconds.append(time.perf_counter() - start_time)
            # TP replicates hidden states and gathers lm_head, so every rank holds the
            # same NLL; only rank 0 keeps it.
            if RANK == 0:
                arrays[f"nll/{label}/doc{i}"] = nll.float().cpu().numpy()
            if method == "probe":
                # collective (all-gathers TP-sharded heads): every rank participates
                for key, arr in probe_state.finish_doc().items():
                    if RANK == 0:
                        arrays[f"probe/{key}/doc{i}"] = arr
        torch.cuda.empty_cache()

        peak_gb = broadcast_max(torch.cuda.max_memory_allocated() / 1e9, device)
        if RANK != 0:
            continue
        mean_nll = float(np.mean([arrays[f"nll/{label}/doc{i}"].mean() for i in range(len(docs))]))
        rows.append(
            {
                "method": method,
                "budget": budget,
                "kept_fraction": equivalent_fraction(budget, args.fp4_speedup) if method.endswith("_drop") else budget,
                "label": label,
                "mean_nll": mean_nll,
                "ppl": math.exp(mean_nll) if mean_nll < 50 else float("inf"),
                "forward_s": fmean(seconds),
                "peak_gb": peak_gb,
            }
        )
        log(f"{label:>14}: nll={mean_nll:.4f}  {fmean(seconds):.1f}s/doc  peak={peak_gb:.1f}GB/gpu")
        if method == "probe":
            rows[-1]["probe"] = summarize_probe(arrays, budgets, len(docs), rows)
            for entry in rows[-1]["probe"]:
                removed = f"  err removed={entry['err_removed']:.3f}" if "err_removed" in entry else ""
                recovered = f"  nll recovery={entry['nll_recovery']:.3f}" if "nll_recovery" in entry else ""
                log(
                    f"{'':>14}  @{entry['budget'] * 100:g}%: err capture={entry['err_capture']:.3f}"
                    f"{removed}{recovered}  mass capture={entry['mass_capture']:.3f}"
                )

    if RANK != 0:
        dist.destroy_process_group()
        return

    # per-token deltas vs the fp16 baseline
    base = [arrays.get(f"nll/{BASELINE}/doc{i}") for i in range(len(docs))]
    for row in rows:
        if row["label"] == BASELINE or base[0] is None:
            continue
        delta = np.concatenate([arrays[f"nll/{row['label']}/doc{i}"] - base[i] for i in range(len(docs))])
        row["delta_mean"] = float(delta.mean())
        row["delta_p50"] = float(np.percentile(delta, 50))
        row["delta_p99"] = float(np.percentile(delta, 99))
        row["delta_max"] = float(delta.max())

    columns = ["label", "mean_nll", "ppl", "delta_mean", "delta_p50", "delta_p99", "delta_max", "forward_s"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        cells = [f"{v:.4g}" if isinstance(v, float) else str(v) for v in (row.get(c, "-") for c in columns)]
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    log("\n" + table)

    np.savez_compressed(out_dir / "per_token_nll.npz", **arrays)
    with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    heading = f"# Per-token NLL ({args.model}, {args.length} tokens, tp={WORLD_SIZE})"
    (out_dir / "summary.md").write_text(f"{heading}\n\n{table}\n", encoding="utf-8")
    log(f"\nWrote {out_dir}")
    if WORLD_SIZE > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
