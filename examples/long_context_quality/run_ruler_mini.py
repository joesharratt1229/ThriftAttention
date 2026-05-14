#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import fmean
from typing import Any


EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from common import (  # noqa: E402
    collect_environment,
    load_config,
    make_output_dir,
    markdown_table,
    parse_int_list,
    parse_str_list,
    set_seed,
    thrift_acceleration_status,
    timed_call,
    write_json,
    write_jsonl,
)
from common.cli import pick  # noqa: E402


DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_TASKS = "needle,variable_tracking,common_words"
DEFAULT_METHODS = "fp16,fp4,thrift"
VALID_TASKS = {"needle", "variable_tracking", "common_words"}
VALID_METHODS = {"fp16": "fp16_flash", "flash": "fp16_flash", "fp16_flash": "fp16_flash", "fp4": "fp4", "thrift": "thrift"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tiny synthetic RULER-style generation benchmark for long-context smoke testing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", default="quick", choices=["quick"])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--lengths", default=None)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    config_path = args.config or Path(__file__).parent / "configs" / "ruler_quick.yaml"
    config = load_config(config_path)
    args.config_path = str(config_path)
    args.model = pick("model", args.model, config, DEFAULT_MODEL, str)
    args.lengths = pick("lengths", args.lengths, config, [4096, 8192], parse_int_list)
    args.tasks = [task.strip() for task in pick("tasks", args.tasks, config, DEFAULT_TASKS, parse_str_list)]
    args.methods = [_normalise_method(method) for method in pick("methods", args.methods, config, DEFAULT_METHODS, parse_str_list)]
    args.fraction = pick("fraction", args.fraction, config, 0.05, float)
    args.num_examples = pick("num_examples", args.num_examples, config, 1, int)
    args.max_new_tokens = pick("max_new_tokens", args.max_new_tokens, config, 24, int)
    args.output = Path(pick("output", args.output, config, Path("results/long_context_quality"), Path))
    args.seed = pick("seed", args.seed, config, 1234, int)
    args.device = pick("device", args.device, config, "cuda", str)
    unknown = sorted(set(args.tasks) - VALID_TASKS)
    if unknown:
        raise SystemExit(f"unknown task(s): {', '.join(unknown)}")
    return args


def _normalise_method(method: str) -> str:
    key = method.lower().strip()
    if key not in VALID_METHODS:
        raise SystemExit(f"unknown method {method!r}; choose from fp16, fp4, thrift")
    return VALID_METHODS[key]


def require_transformers() -> None:
    try:
        __import__("transformers")
    except Exception:
        raise SystemExit(
            "Missing Transformers. Install with `pip install -r examples/long_context_quality/requirements.txt` "
            "or `pip install -e '.[hf]'`."
        )


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available. Use `--device cpu` for fp16-only smoke runs.")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = None
    if args.device.startswith("cuda"):
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map={"": args.device})
        except Exception as exc:
            print(f"device_map load failed ({exc}); retrying with explicit .to({args.device}).")
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
        model.to(args.device)
    model.eval()
    return model, tokenizer


def configure_method(model: Any, method: str, args: argparse.Namespace, thrift_ready: bool, thrift_note: str) -> tuple[bool, str]:
    import thriftattention as ta

    try:
        ta.unpatch_model(model, backend="hf")
    except Exception:
        pass
    if method == "fp16_flash":
        return True, "using model's standard Transformers attention implementation"
    if not thrift_ready:
        return False, thrift_note
    try:
        ta.patch_model(model, backend="hf", mode="fp4" if method == "fp4" else "thrift", causal=True, fp16_fraction=args.fraction)
    except Exception as exc:
        return False, f"could not enable {method}: {exc}"
    return True, f"enabled {method}"


def build_task_input(tokenizer: Any, task: str, length: int, seed: int) -> tuple[list[int], str]:
    import random

    rng = random.Random(seed)
    if task == "needle":
        answer = f"K{rng.randrange(100000, 999999)}"
        head = f"A useful fact is hidden here. The retrieval key is {answer}. Remember this exact key.\n"
        tail = "\nQuestion: What is the retrieval key? Answer with only the key.\nAnswer:"
    elif task == "variable_tracking":
        choices = ["cerulean", "magenta", "amber", "teal"]
        answer = choices[seed % len(choices)]
        head = f"Initialize variables. alpha = {answer}. beta = slate. gamma = copper. Later alpha keeps its original value.\n"
        tail = "\nQuestion: What is the final value of alpha? Answer with one word.\nAnswer:"
    elif task == "common_words":
        answer = "apple"
        head = "Word multiset: apple apple apple pear pear plum. Count the words carefully.\n"
        tail = "\nQuestion: Which word appears most often in the multiset? Answer with one word.\nAnswer:"
    else:
        raise ValueError(task)

    filler = (
        " Background sentence for long-context padding. It is unrelated to the answer and should be ignored."
    )
    return _pack_to_length(tokenizer, head, filler, tail, length), answer


def _pack_to_length(tokenizer: Any, head: str, filler: str, tail: str, length: int) -> list[int]:
    head_ids = tokenizer(head, add_special_tokens=False)["input_ids"]
    tail_ids = tokenizer(tail, add_special_tokens=False)["input_ids"]
    filler_ids = tokenizer(filler, add_special_tokens=False)["input_ids"] or [tokenizer.eos_token_id or 0]
    budget = max(0, length - len(head_ids) - len(tail_ids))
    middle = (filler_ids * ((budget // len(filler_ids)) + 1))[:budget]
    ids = head_ids + middle + tail_ids
    if len(ids) > length:
        ids = ids[: max(0, length - len(tail_ids))] + tail_ids
    return ids


def generate_answer(model: Any, tokenizer: Any, input_ids: list[int], args: argparse.Namespace) -> str:
    import torch

    encoded = torch.tensor([input_ids], dtype=torch.long, device=args.device)
    with torch.inference_mode():
        output = model.generate(
            input_ids=encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, encoded.shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    args = resolve_args(parse_args())
    require_transformers()
    set_seed(args.seed)
    output_dir = make_output_dir(args.output, prefix="ruler-mini")
    write_json(output_dir / "environment.json", collect_environment(args))

    print(f"Writing results to {output_dir}")
    model, tokenizer = load_model_and_tokenizer(args)
    thrift_ready, thrift_note = thrift_acceleration_status(args.device)
    if not thrift_ready:
        print(f"Accelerated methods will be skipped if requested: {thrift_note}")

    rows: list[dict[str, Any]] = []
    for method in args.methods:
        ok, note = configure_method(model, method, args, thrift_ready, thrift_note)
        print(f"\nMethod {method}: {note}")
        if not ok:
            for length in args.lengths:
                for task in args.tasks:
                    rows.append({"method": method, "length": length, "task": task, "status": "skipped", "accuracy": None, "error": note})
            continue
        for length in args.lengths:
            for task in args.tasks:
                correct: list[float] = []
                elapsed_s: list[float] = []
                for index in range(args.num_examples):
                    input_ids, expected = build_task_input(tokenizer, task, length, args.seed + index + length)

                    def run_one() -> str:
                        return generate_answer(model, tokenizer, input_ids, args)

                    try:
                        prediction, elapsed = timed_call(run_one, device=args.device)
                        is_correct = expected.lower() in prediction.lower()
                        correct.append(1.0 if is_correct else 0.0)
                        elapsed_s.append(elapsed)
                        print(
                            f"  {method:<11} length={length:<6} task={task:<18} "
                            f"expected={expected!r} correct={is_correct} wall_s={elapsed:.3f}"
                        )
                    except RuntimeError as exc:
                        rows.append(
                            {
                                "method": method,
                                "length": length,
                                "task": task,
                                "example": index,
                                "status": "error",
                                "accuracy": None,
                                "error": str(exc),
                            }
                        )
                if correct:
                    rows.append(
                        {
                            "method": method,
                            "length": length,
                            "task": task,
                            "status": "ok",
                            "accuracy": fmean(correct),
                            "num_examples": len(correct),
                            "mean_wall_s": fmean(elapsed_s),
                            "fraction": args.fraction if method == "thrift" else None,
                        }
                    )

    write_jsonl(output_dir / "metrics.jsonl", rows)
    summary_rows = _summary_rows(rows)
    summary = "\n".join(
        [
            "# RULER Mini",
            "",
            "Tiny synthetic generation tasks inspired by RULER. This is a smoke test, not an official RULER score.",
            "",
            markdown_table(summary_rows, [("method", "method"), ("length", "length"), ("accuracy", "mean accuracy"), ("status", "status")]),
            "",
            f"Model: `{args.model}`",
            f"Tasks: `{', '.join(args.tasks)}`",
        ]
    )
    (output_dir / "summary.md").write_text(summary + "\n", encoding="utf-8")
    print(f"\nWrote metrics.jsonl, summary.md, and environment.json under {output_dir}")


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["method"]), int(row["length"])), []).append(row)
    summary: list[dict[str, Any]] = []
    for (method, length), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        ok = [float(item["accuracy"]) for item in items if item.get("status") == "ok" and item.get("accuracy") is not None]
        status = "ok" if ok else items[0].get("status", "skipped")
        summary.append({"method": method, "length": length, "accuracy": f"{fmean(ok):.3f}" if ok else "-", "status": status})
    return summary


if __name__ == "__main__":
    main()
