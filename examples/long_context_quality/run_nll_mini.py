#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean


DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_DATASET = "emozilla/pg19"
FLASH_ATTN_IMPLEMENTATION = "flash_attention_2"
METHODS = {
    "fp16": {"method": "fp16", "label": "fp16", "exp_approx": False},
    "fp4": {"method": "fp4", "label": "fp4", "exp_approx": False},
    "fp4_exp": {"method": "fp4", "label": "fp4_exp", "exp_approx": False},
    "fp4_exp_approx": {"method": "fp4", "label": "fp4_exp_approx", "exp_approx": True},
    "fp4_approx": {"method": "fp4", "label": "fp4_exp_approx", "exp_approx": True},
    "thrift": {"method": "thrift", "label": "thrift", "exp_approx": False},
}


def load_text_file_docs(tokenizer, path: Path, *, length: int, num_docs: int) -> list[list[int]]:
    text = path.read_text(encoding="utf-8")
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    required = length * num_docs
    if len(ids) < required:
        raise SystemExit(
            f"{path} only tokenized to {len(ids)} tokens, but this run needs "
            f"{required} tokens ({num_docs} docs x {length})."
        )
    return [ids[i * length : (i + 1) * length] for i in range(num_docs)]


def load_pg19_docs(tokenizer, *, dataset: str, length: int, num_docs: int, seed: int) -> list[list[int]]:
    from datasets import load_dataset

    required = length * num_docs
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.bos_token_id
    buffer: list[int] = []

    print(f"Loading {num_docs} x {length} real tokens from {dataset}")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini forward/NLL benchmark for registered HF attention implementations.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--lengths", default="131072")
    parser.add_argument(
        "--methods",
        default="fp4,fp4_exp_approx",
        help="Comma-separated methods. Use fp4 for standard exp, fp4_exp_approx for approximate exp; fp16 and thrift are opt-in.",
    )
    parser.add_argument("--fractions", default="0.05")
    parser.add_argument("--num-docs", type=int, default=1)
    parser.add_argument("--text-file", type=Path, default=None)
    parser.add_argument("--ce-chunk", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=Path("results/long_context_quality"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from thriftattention.integrations.transformers import (
        TransformersAttentionConfig,
        register_transformers_attention,
    )
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    lengths = [int(x) for x in args.lengths.replace(" ", ",").split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    fractions = [float(x) for x in args.fractions.replace(" ", ",").split(",") if x.strip()]

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = None
    if args.output:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out_dir = args.output / f"{stamp}-nll-mini"
        out_dir.mkdir(parents=True)
        (out_dir / "environment.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True, default=str) + "\n")

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(args.model)
    if max(lengths) > 32768:
        max_pos = max(lengths)
        original = 32768
        config.max_position_embeddings = max_pos
        if getattr(config, "rope_theta", None) is None:
            config.rope_theta = 1000000.0
        config.rope_scaling = {
            "rope_type": "yarn",
            "factor": max_pos / original,
            "original_max_position_embeddings": original,
        }
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=dtype,
        attn_implementation=FLASH_ATTN_IMPLEMENTATION,
    ).to(args.device).eval()

    max_length = max(lengths)
    docs = (
        load_text_file_docs(tokenizer, args.text_file, length=max_length, num_docs=args.num_docs)
        if args.text_file
        else load_pg19_docs(
            tokenizer,
            dataset=args.dataset,
            length=max_length,
            num_docs=args.num_docs,
            seed=args.seed,
        )
    )

    body = getattr(model, getattr(model, "base_model_prefix", ""), None)
    for name in ("model", "transformer", "gpt_neox", "backbone"):
        body = body or getattr(model, name, None)

    rows = []
    for raw_method in methods:
        try:
            method_spec = METHODS[raw_method.lower()]
        except KeyError as exc:
            choices = ", ".join(sorted(METHODS))
            raise SystemExit(f"unknown method {raw_method!r}; choose from: {choices}") from exc

        method = method_spec["method"]
        exp_approx = bool(method_spec["exp_approx"])
        run_fractions = fractions if method == "thrift" else [None]

        for fraction in run_fractions:
            label = (
                f"thrift_{fraction * 100:g}pct".replace(".", "p")
                if method == "thrift"
                else method_spec["label"]
            )

            if method == "fp16":
                model.set_attn_implementation(FLASH_ATTN_IMPLEMENTATION)
                print(f"\n{label}: fp16 FlashAttention 2")
            else:
                if method == "fp4":
                    impl_name = "thrift_fp4_exp_approx" if exp_approx else f"thrift_{label}"
                else:
                    impl_name = f"thrift_attention_{fraction * 100:g}pct".replace(".", "p")
                attn_impl = register_transformers_attention(
                    TransformersAttentionConfig(
                        name=impl_name,
                        method="fp4" if method == "fp4" else "thrift",
                        fraction=0.0 if fraction is None else fraction,
                        exp_approx=exp_approx,
                    )
                )
                model.set_attn_implementation(attn_impl)
                print(f"\n{label}: registered {attn_impl}")

            for length in lengths:
                values, seconds = [], []
                for doc in docs:
                    input_ids = torch.tensor([doc[:length]], dtype=torch.long, device=args.device)
                    targets = torch.tensor(doc[1:length], dtype=torch.long, device=args.device)
                    autocast = torch.autocast("cuda", dtype=dtype) if args.device.startswith("cuda") else nullcontext()

                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize(args.device)
                    start_time = time.perf_counter()
                    with torch.inference_mode(), autocast:
                        hidden = body(input_ids=input_ids, use_cache=False, return_dict=True).last_hidden_state[:, :-1, :]
                        total = 0.0
                        for start in range(0, targets.numel(), args.ce_chunk):
                            end = min(start + args.ce_chunk, targets.numel())
                            logits = model.get_output_embeddings()(hidden[:, start:end, :]).squeeze(0).float()
                            total += float(F.cross_entropy(logits, targets[start:end], reduction="sum"))
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize(args.device)

                    values.append(total / targets.numel())
                    seconds.append(time.perf_counter() - start_time)

                mean_nll = fmean(values)
                forward_s = fmean(seconds)
                row = {
                    "method": method,
                    "label": label,
                    "fraction": fraction,
                    "exp_approx": exp_approx,
                    "length": length,
                    "status": "ok",
                    "mean_nll": mean_nll,
                    "ppl": math.exp(mean_nll) if mean_nll < 50 else float("inf"),
                    "forward_s": forward_s,
                    "forward_tok_s": length / forward_s,
                }
                rows.append(row)
                print(f"  length={length:<6} nll={mean_nll:.4f} forward_s={forward_s:.3f}")

    fp16_baselines = {row["length"]: row["mean_nll"] for row in rows if row["label"] == "fp16"}
    fp4_baselines = {
        row["length"]: row["mean_nll"]
        for row in rows
        if row["method"] == "fp4" and not row["exp_approx"]
    }
    for row in rows:
        if row["length"] in fp4_baselines:
            row["delta_vs_fp4"] = row["mean_nll"] - fp4_baselines[row["length"]]
        if row["length"] in fp16_baselines:
            row["delta_vs_fp16"] = row["mean_nll"] - fp16_baselines[row["length"]]

    columns = [
        ("length", "tokens"),
        ("label", "method"),
        ("mean_nll", "mean_nll"),
        ("delta_vs_fp4", "delta_fp4"),
        ("delta_vs_fp16", "delta_fp16"),
        ("forward_s", "forward_s"),
        ("forward_tok_s", "tok/s"),
        ("status", "status"),
    ]
    lines = ["| " + " | ".join(label for _, label in columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if value is None:
                value = "-"
            elif isinstance(value, float):
                value = "inf" if value == float("inf") else f"{value:.4g}"
            cells.append(str(value).replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)

    print("\nAverage scores")
    print(table)

    if out_dir:
        with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        (out_dir / "summary.md").write_text("# NLL Mini\n\n" + table + "\n", encoding="utf-8")
        print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
