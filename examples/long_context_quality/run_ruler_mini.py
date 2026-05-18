#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

from dataset_utils import ruler_gen, ruler_score


DEFAULT_MODEL = "Qwen/Qwen3-8B"
FLASH_ATTN_IMPLEMENTATION = "flash_attention_2"
METHODS = {
    "fp16": "fp16",
    "fp4": "fp4",
    "thrift": "thrift",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini RULER generation benchmark.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--lengths", default="32768")
    parser.add_argument("--tasks", default="niah_multikey_3")
    parser.add_argument("--methods", default="fp16,fp4,thrift")
    parser.add_argument("--fractions", default="0.05")
    parser.add_argument("--num-examples", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    from thriftattention.integrations.transformers import (
        TransformersAttentionConfig,
        prepare_transformers_generation_cache,
        register_transformers_attention,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    lengths = [int(x) for x in args.lengths.replace(" ", ",").split(",") if x.strip()]
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    fractions = [float(x) for x in args.fractions.replace(" ", ",").split(",") if x.strip()]

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = None
    if args.output:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out_dir = args.output / f"{stamp}-ruler-mini"
        out_dir.mkdir(parents=True)
        (out_dir / "environment.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True, default=str) + "\n")

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation=FLASH_ATTN_IMPLEMENTATION,
    ).to(args.device).eval()

    rows = []
    for raw_method in methods:
        method = METHODS[raw_method.lower()]
        run_fractions = fractions if method == "thrift" else [None]

        for fraction in run_fractions:
            config = None
            label = f"thrift_{fraction * 100:g}pct".replace(".", "p") if method == "thrift" else method

            if method == "fp16":
                model.set_attn_implementation(FLASH_ATTN_IMPLEMENTATION)
                print(f"\n{label}: fp16 FlashAttention 2")
            else:
                impl_name = "thrift_fp4" if method == "fp4" else f"thrift_attention_{fraction * 100:g}pct".replace(".", "p")
                config = TransformersAttentionConfig(
                    name=impl_name,
                    mode="fp4" if method == "fp4" else "thrift",
                    fp16_fraction=0.0 if fraction is None else fraction,
                )
                attn_impl = register_transformers_attention(config)
                model.set_attn_implementation(attn_impl)
                print(f"\n{label}: registered {attn_impl}")

            for length in lengths:
                for task in tasks:
                    samples = ruler_gen.generate_samples(tokenizer, task, length, args.num_examples, args.seed, args.cache_dir)

                    for index, sample in enumerate(samples):
                        prompt = sample["input"] + sample["answer_prefix"]
                        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
                        max_new_tokens = int(args.max_new_tokens or sample["max_gen"])
                        prompt_len = len(input_ids)

                        cache_inputs = None
                        if config is not None:
                            cache_inputs = prepare_transformers_generation_cache(
                                model,
                                input_ids,
                                config=config,
                                max_new_tokens=max_new_tokens,
                                device=args.device,
                            )
                            encoded = cache_inputs.input_ids
                            cache_position = cache_inputs.cache_position
                            past = cache_inputs.past_key_values
                            cache_ctx = cache_inputs.activate()
                        else:
                            encoded = torch.tensor([input_ids], dtype=torch.long, device=args.device)
                            cache_position = torch.arange(encoded.shape[1], device=args.device, dtype=torch.long)
                            past = None
                            cache_ctx = nullcontext()

                        with torch.inference_mode(), cache_ctx:
                            if args.device.startswith("cuda"):
                                torch.cuda.synchronize(args.device)
                            start_time = time.perf_counter()
                            out = model(input_ids=encoded, use_cache=True, past_key_values=past, cache_position=cache_position)
                            if args.device.startswith("cuda"):
                                torch.cuda.synchronize(args.device)
                            prefill_s = time.perf_counter() - start_time

                            past = out.past_key_values
                            if cache_inputs is not None:
                                cache_inputs.trim_padding()

                            next_token = out.logits[:, prompt_len - 1, :].argmax(dim=-1)
                            generated = []

                            if args.device.startswith("cuda"):
                                torch.cuda.synchronize(args.device)
                            start_time = time.perf_counter()
                            for step in range(max_new_tokens):
                                token = int(next_token.item())
                                generated.append(token)
                                if token == tokenizer.eos_token_id or step == max_new_tokens - 1:
                                    break
                                cache_position = torch.tensor([prompt_len + step], device=args.device, dtype=torch.long)
                                out = model(input_ids=next_token[:, None], use_cache=True, past_key_values=past, cache_position=cache_position)
                                past = out.past_key_values
                                next_token = out.logits[:, -1, :].argmax(dim=-1)
                            if args.device.startswith("cuda"):
                                torch.cuda.synchronize(args.device)
                            decode_s = time.perf_counter() - start_time

                        decode_steps = max(0, len(generated) - 1)
                        prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
                        accuracy = float(ruler_score.score_sample(task, prediction, sample["outputs"]))
                        row = {
                            "method": method,
                            "label": label,
                            "fraction": fraction,
                            "length": length,
                            "sample_length": sample.get("length"),
                            "task": task,
                            "example": index,
                            "status": "ok",
                            "outputs": sample["outputs"],
                            "prediction": prediction,
                            "accuracy": accuracy,
                            "prompt_tokens": prompt_len,
                            "generated_tokens": len(generated),
                            "decode_steps": decode_steps,
                            "prefill_s": prefill_s,
                            "decode_s": decode_s,
                            "total_s": prefill_s + decode_s,
                            "decode_tok_s": decode_steps / decode_s if decode_s > 0 else None,
                            "e2e_tok_s": (prompt_len + len(generated)) / (prefill_s + decode_s),
                        }
                        rows.append(row)
                        print(f"  len={length:<6} task={task:<17} acc={accuracy:.3f} prefill={prefill_s:.3f}s decode={decode_s:.3f}s total={row['total_s']:.3f}s")

    grouped = {}
    for row in rows:
        grouped.setdefault((row["length"], row["task"], row["label"]), []).append(row)

    summary = []
    for (length, task, label), items in sorted(grouped.items()):
        summary.append(
            {
                "tokens": length,
                "task": task,
                "method": label,
                "score": f"{fmean(item['accuracy'] for item in items):.3f}",
                "prefill_s": f"{fmean(item['prefill_s'] for item in items):.3f}",
                "decode_s": f"{fmean(item['decode_s'] for item in items):.3f}",
                "total_s": f"{fmean(item['total_s'] for item in items):.3f}",
                "n": len(items),
                "status": "ok",
            }
        )

    columns = [
        ("tokens", "tokens"),
        ("task", "task"),
        ("method", "method"),
        ("score", "avg_score"),
        ("prefill_s", "prefill_s"),
        ("decode_s", "decode_s"),
        ("total_s", "total_s"),
        ("n", "n"),
        ("status", "status"),
    ]
    lines = ["| " + " | ".join(label for _, label in columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in summary:
        lines.append("| " + " | ".join(str(row.get(key, "")).replace("\n", " ") for key, _ in columns) + " |")
    table = "\n".join(lines)

    print("\nAverage scores")
    print(table)

    if out_dir:
        with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        (out_dir / "summary.md").write_text("# RULER Mini\n\n" + table + "\n", encoding="utf-8")
        print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
