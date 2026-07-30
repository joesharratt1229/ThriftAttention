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

Budgets apply to the thrift/drop methods only. Smoke test on one RTX 6000 Pro:
  python run_nll_per_token.py --length 16384
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from statistics import fmean

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-token NLL for ThriftAttention methods on long-context text.")
    parser.add_argument("--model", default="ByteDance-Seed/Seed-OSS-36B-Base")
    parser.add_argument("--dataset", default="emozilla/pg19")
    parser.add_argument("--length", type=int, default=16384)
    parser.add_argument(
        "--methods",
        default="fp16,local,quest,block_mean,quest_drop,block_mean_drop,fp4",
        help="comma list of fp16, local, quest, block_mean, quest_drop, block_mean_drop, local_drop, fp4",
    )
    parser.add_argument("--budgets", default="0.05,0.10,0.25", help="fp16 block budgets for the thrift methods")
    parser.add_argument(
        "--fp4-speedup",
        type=float,
        default=4.0,
        help="assumed fp4:fp16 attention throughput ratio used to compute-match the *_drop budgets; inf runs them at --budgets as-is",
    )
    parser.add_argument("--num-docs", type=int, default=1)
    parser.add_argument("--ce-chunk", type=int, default=1024)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--baseline-impl", default="flash_attention_2", help="use sdpa if flash-attn is unavailable")
    parser.add_argument("--output", type=Path, default=Path("results/nll_per_token"))
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def load_pg19_docs(tokenizer, *, dataset: str, length: int, num_docs: int, seed: int) -> list[list[int]]:
    from datasets import load_dataset

    required = length * num_docs
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.bos_token_id
    buffer: list[int] = []

    print(f"Loading {num_docs} x {length} tokens from {dataset}")
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


def build_runs(methods: list[str], budgets: list[float]) -> list[tuple[str, float | None]]:
    runs: list[tuple[str, float | None]] = []
    for method in methods:
        if method in ("fp16", "fp4"):
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


def attention_impl(method: str, budget: float | None, args: argparse.Namespace) -> str:
    if method == BASELINE:
        return args.baseline_impl
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


@torch.no_grad()  # not inference_mode: torch.compile'd flex_attention rejects inference tensors
def per_token_nll(model, body, doc: list[int], ce_chunk: int) -> torch.Tensor:
    """NLL of tokens 1..n-1 from a single forward pass, lm_head applied in chunks."""
    input_ids = torch.tensor([doc], dtype=torch.long, device=model.device)
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
        print(f"drop methods keep fractions [{kept}] (compute-equivalent to {budgets} at {args.fp4_speedup:g}x fp4 speedup)")
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    docs = load_pg19_docs(tokenizer, dataset=args.dataset, length=args.length, num_docs=args.num_docs, seed=args.seed)

    print(f"Loading {args.model} in {args.dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=getattr(torch, args.dtype),
        device_map="cuda",
        attn_implementation=args.baseline_impl,
    ).eval()
    body = getattr(model, model.base_model_prefix)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = args.output / f"{stamp}-{args.model.rsplit('/', 1)[-1]}-{args.length}"
    out_dir.mkdir(parents=True)
    (out_dir / "environment.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")

    arrays = {f"tokens/doc{i}": np.asarray(doc, dtype=np.int32) for i, doc in enumerate(docs)}
    rows = []
    for method, budget in runs:
        label = run_label(method, budget)
        model.set_attn_implementation(attention_impl(method, budget, args))
        torch.cuda.reset_peak_memory_stats()
        seconds = []
        for i, doc in enumerate(docs):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            nll = per_token_nll(model, body, doc, args.ce_chunk)
            torch.cuda.synchronize()
            seconds.append(time.perf_counter() - start_time)
            arrays[f"nll/{label}/doc{i}"] = nll.float().cpu().numpy()
        torch.cuda.empty_cache()

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
                "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
            }
        )
        print(f"{label:>14}: nll={mean_nll:.4f}  {fmean(seconds):.1f}s/doc")

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
    print("\n" + table)

    np.savez_compressed(out_dir / "per_token_nll.npz", **arrays)
    with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (out_dir / "summary.md").write_text(f"# Per-token NLL ({args.model}, {args.length} tokens)\n\n{table}\n", encoding="utf-8")
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
