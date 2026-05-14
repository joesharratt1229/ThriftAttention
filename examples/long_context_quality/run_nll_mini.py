#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np


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
    write_json,
    write_jsonl,
)
from common.cli import pick  # noqa: E402


DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_DATASET = "pg19"
DEFAULT_LENGTHS = "8192,32768,65536"
DEFAULT_METHODS = "fp16,fp4,thrift"
DEFAULT_FRACTIONS = "0.05,0.10,0.25"
BLOCK_SIZE = 64
VALID_METHODS = {
    "fp16": "fp16",
    "flash": "fp16",
    "fp16_flash": "fp16",
    "sdpa": "fp16",
    "fp4": "fp4",
    "thrift": "thrift",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mini NLL-over-context-length demo for ThriftAttention.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=None, help="Hugging Face causal LM id.")
    parser.add_argument("--dataset", default=None, help="HF dataset name, or 'synthetic' for an offline smoke input.")
    parser.add_argument("--lengths", default=None, help="Comma-separated context lengths.")
    parser.add_argument("--methods", default=None, help="Comma-separated methods: fp16, fp4, thrift.")
    parser.add_argument("--fractions", default=None, help="Comma-separated FP16 block fractions for thrift mode.")
    parser.add_argument("--fraction", type=float, default=None, help="Backward-compatible single thrift budget.")
    parser.add_argument("--preset", default="quick", choices=["quick", "standard"], help="Config preset to load.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config override.")
    parser.add_argument("--num-docs", type=int, default=None, help="Number of long text chunks to score.")
    parser.add_argument("--ce-chunk", type=int, default=None, help="LM-head chunk size for per-token cross entropy.")
    parser.add_argument("--output", type=Path, default=None, help="Output root; a timestamped run directory is created inside it.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    preset_path = Path(__file__).parent / "configs" / f"nll_{args.preset}.yaml"
    config = load_config(args.config or preset_path)
    args.config_path = str(args.config or preset_path)
    args.model = pick("model", args.model, config, DEFAULT_MODEL, str)
    args.dataset = pick("dataset", args.dataset, config, DEFAULT_DATASET, str)
    args.dataset_split = str(config.get("dataset_split", "test"))
    args.lengths = pick("lengths", args.lengths, config, DEFAULT_LENGTHS, parse_int_list)
    args.methods = [_normalise_method(method) for method in pick("methods", args.methods, config, DEFAULT_METHODS, parse_str_list)]
    raw_fractions = (
        args.fractions
        if args.fractions is not None
        else config.get(
            "fractions",
            args.fraction if args.fraction is not None else config.get("fraction", DEFAULT_FRACTIONS),
        )
    )
    args.fractions = parse_float_list(raw_fractions)
    args.num_docs = pick("num_docs", args.num_docs, config, 1, int)
    args.ce_chunk = pick("ce_chunk", args.ce_chunk, config, 2048, int)
    args.output = Path(pick("output", args.output, config, Path("results/long_context_quality"), Path))
    args.seed = pick("seed", args.seed, config, 1234, int)
    args.device = pick("device", args.device, config, "cuda", str)
    if args.num_docs < 1:
        raise SystemExit("--num-docs must be at least 1")
    if args.ce_chunk < 1:
        raise SystemExit("--ce-chunk must be at least 1")
    if any(not 0.0 <= fraction <= 1.0 for fraction in args.fractions):
        raise SystemExit("--fractions must all be in [0, 1]")
    if any(length <= 0 for length in args.lengths):
        raise SystemExit("--lengths must all be positive")
    if any(length % BLOCK_SIZE != 0 for length in args.lengths):
        raise SystemExit(f"--lengths must be multiples of {BLOCK_SIZE} for fp4/thrift kernels")
    return args


def _normalise_method(method: str) -> str:
    key = method.strip().lower()
    if key not in VALID_METHODS:
        valid = ", ".join(sorted(VALID_METHODS))
        raise SystemExit(f"unknown method {method!r}; expected one of {valid}")
    return VALID_METHODS[key]


def parse_float_list(value: float | str | list[float] | tuple[float, ...]) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    items = [float(part.strip()) for part in str(value).replace(" ", ",").split(",") if part.strip()]
    if not items:
        raise ValueError("expected at least one float")
    return items


def require_hf_stack(args: argparse.Namespace) -> None:
    missing: list[str] = []
    modules = ["transformers"]
    if args.dataset.lower() not in {"synthetic", "offline"}:
        modules.append("datasets")
    for module in modules:
        try:
            __import__(module)
        except Exception:
            missing.append(module)
    if missing:
        raise SystemExit(
            "Missing optional dependencies for the NLL example: "
            + ", ".join(missing)
            + ". Install with `pip install -r examples/long_context_quality/requirements.txt` "
            "or `pip install -e '.[hf,plots]'`."
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
            model = _from_pretrained(AutoModelForCausalLM, args.model, dtype=dtype, device_map={"": args.device})
        except Exception as exc:
            print(f"device_map load failed ({exc}); retrying with explicit .to({args.device}).")
    if model is None:
        model = _from_pretrained(AutoModelForCausalLM, args.model, dtype=dtype)
        model.to(args.device)
    model.eval()
    return model, tokenizer


def _from_pretrained(model_cls: Any, model_id: str, **kwargs: Any) -> Any:
    try:
        return model_cls.from_pretrained(model_id, **kwargs)
    except TypeError:
        if "dtype" not in kwargs:
            raise
        fallback = dict(kwargs)
        fallback["torch_dtype"] = fallback.pop("dtype")
        return model_cls.from_pretrained(model_id, **fallback)


def load_token_documents(tokenizer: Any, args: argparse.Namespace, max_length: int) -> list[list[int]]:
    if args.dataset.lower() in {"synthetic", "offline"}:
        return list(_synthetic_documents(tokenizer, max_length=max_length, count=args.num_docs))
    return list(_dataset_documents(tokenizer, args, max_length=max_length))


def _synthetic_documents(tokenizer: Any, *, max_length: int, count: int) -> Iterable[list[int]]:
    seed_text = (
        "ThriftAttention evaluates long-context next-token likelihood with repeated technical prose. "
        "The text is intentionally plain so the example can run without downloading a dataset. "
    )
    token_ids = tokenizer(seed_text, add_special_tokens=False)["input_ids"]
    if not token_ids:
        token_ids = [tokenizer.eos_token_id or 0]
    for index in range(count):
        rotated = token_ids[index % len(token_ids) :] + token_ids[: index % len(token_ids)]
        repeats = math.ceil((max_length + 1) / len(rotated))
        yield (rotated * repeats)[: max_length + 1]


def _dataset_documents(tokenizer: Any, args: argparse.Namespace, *, max_length: int) -> Iterable[list[int]]:
    from datasets import load_dataset

    candidates = [args.dataset]
    if args.dataset.lower() in {"pg19", "pg-19"}:
        candidates = ["emozilla/pg19", "fla-hub/pg19", "pg19", "deepmind/pg19"]

    errors: list[str] = []
    dataset = None
    for name in candidates:
        try:
            dataset = load_dataset(name, split=args.dataset_split, streaming=True)
            break
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    if dataset is None:
        raise SystemExit(
            "Could not load the requested dataset. Tried:\n  "
            + "\n  ".join(errors)
            + "\nUse `--dataset synthetic` for an offline smoke run."
        )

    buffer: list[int] = []
    eos = tokenizer.eos_token_id
    produced = 0
    for example in dataset:
        text = _extract_text(example)
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        buffer.extend(ids)
        if eos is not None:
            buffer.append(eos)
        while len(buffer) >= max_length + 1 and produced < args.num_docs:
            yield buffer[: max_length + 1]
            produced += 1
            buffer = buffer[max_length + 1 :]
        if produced >= args.num_docs:
            return
    raise SystemExit(
        f"Dataset {args.dataset!r} did not provide {args.num_docs} chunks with "
        f"{max_length + 1} tokens. Try fewer docs, shorter lengths, or another dataset."
    )


def _extract_text(example: dict[str, Any]) -> str:
    for key in ("text", "content", "document", "article"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for value in example.values():
        if isinstance(value, str) and value.strip():
            return value
    return ""


def configure_method(
    model: Any,
    method: str,
    *,
    fraction: float | None,
    thrift_ready: bool,
    thrift_note: str,
) -> tuple[bool, str]:
    import thriftattention as ta

    try:
        ta.unpatch_model(model, backend="hf")
    except Exception:
        pass

    if method == "fp16":
        return True, "using model's standard Transformers attention implementation"

    if not thrift_ready:
        return False, thrift_note

    try:
        ta.patch_model(
            model,
            backend="hf",
            mode="fp4" if method == "fp4" else "thrift",
            causal=True,
            fp16_fraction=0.0 if fraction is None else fraction,
            patch_generation=False,
        )
    except Exception as exc:
        return False, f"could not enable {method}: {exc}"
    if method == "thrift":
        return True, f"enabled thrift at fp16_fraction={fraction:g}"
    return True, "enabled fp4"


def per_token_nll(model: Any, ids: list[int], length: int, args: argparse.Namespace) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    if len(ids) < length:
        raise ValueError(f"document has {len(ids)} tokens, need {length}")

    input_ids = torch.tensor([ids[:length]], dtype=torch.long, device=args.device)
    targets = torch.tensor(ids[1:length], dtype=torch.long, device=args.device)
    body = transformer_body(model)
    lm_head = model.get_output_embeddings()
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    autocast = torch.autocast("cuda", dtype=dtype) if args.device.startswith("cuda") else nullcontext()

    with torch.inference_mode(), autocast:
        hidden = body(input_ids=input_ids, use_cache=False, return_dict=True).last_hidden_state
        hidden = hidden[:, :-1, :]
        out = torch.empty(targets.numel(), dtype=torch.float32, device=args.device)
        for start in range(0, targets.numel(), args.ce_chunk):
            end = min(start + args.ce_chunk, targets.numel())
            logits = lm_head(hidden[:, start:end, :]).squeeze(0).float()
            out[start:end] = F.cross_entropy(logits, targets[start:end], reduction="none")

    result = out.cpu().numpy()
    del input_ids, targets, hidden, out
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def transformer_body(model: Any) -> Any:
    prefix = getattr(model, "base_model_prefix", None)
    if prefix and hasattr(model, prefix):
        return getattr(model, prefix)
    for name in ("model", "transformer", "gpt_neox", "backbone"):
        if hasattr(model, name):
            return getattr(model, name)
    raise RuntimeError(
        "Could not find the transformer body. This NLL path expects a HF causal LM "
        "with a callable base model and get_output_embeddings()."
    )


def method_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for method in args.methods:
        if method == "thrift":
            for fraction in args.fractions:
                runs.append({"method": method, "label": fraction_label(fraction), "fraction": fraction})
        else:
            runs.append({"method": method, "label": method, "fraction": None})
    return runs


def fraction_label(fraction: float) -> str:
    pct = fraction * 100.0
    return f"thrift_{f'{pct:g}'.replace('.', 'p')}pct"


def run_cell(
    model: Any,
    docs: list[list[int]],
    run: dict[str, Any],
    length: int,
    args: argparse.Namespace,
    npz_dir: Path,
) -> dict[str, Any]:
    per_doc: list[np.ndarray] = []
    doc_means: list[float] = []
    label = str(run["label"])

    for doc_index, ids in enumerate(docs, start=1):
        nlls = per_token_nll(model, ids, length, args)
        per_doc.append(nlls.astype(np.float32, copy=False))
        doc_means.append(float(nlls.mean()))
        print(
            f"  {label:<13} length={length:<7} doc={doc_index}/{len(docs)} "
            f"mean_nll={doc_means[-1]:.4f}",
            flush=True,
        )

    mean_nll = float(sum(doc_means) / len(doc_means))
    npz_name = f"{label}_len{length}.npz"
    np.savez_compressed(
        npz_dir / npz_name,
        per_token_nll=np.stack(per_doc, axis=0),
        doc_mean_nll=np.array(doc_means, dtype=np.float32),
    )
    return {
        "label": label,
        "method": run["method"],
        "fraction": run["fraction"],
        "length": length,
        "status": "ok",
        "mean_nll": mean_nll,
        "ppl": math.exp(mean_nll) if mean_nll < 50 else float("inf"),
        "num_docs": len(doc_means),
        "per_token_npz": f"per_token/{npz_name}",
    }


def add_deltas(rows: list[dict[str, Any]]) -> None:
    fp16_by_length = {
        row["length"]: row["mean_nll"]
        for row in rows
        if row.get("status") == "ok" and row.get("label") == "fp16"
    }
    for row in rows:
        baseline = fp16_by_length.get(row["length"])
        if baseline is not None and row.get("mean_nll") is not None:
            row["delta_vs_fp16"] = row["mean_nll"] - baseline


def build_summary(rows: list[dict[str, Any]], lengths: list[int], labels: list[str]) -> str:
    table_rows: list[dict[str, str]] = []
    for length in lengths:
        summary_row: dict[str, str] = {"length": f"{length:,}"}
        for label in labels:
            row = next(
                (item for item in rows if item["length"] == length and item["label"] == label),
                None,
            )
            if row is None:
                summary_row[label] = "-"
            elif row.get("status") == "ok":
                delta = row.get("delta_vs_fp16")
                suffix = "" if delta is None else f" ({delta:+.3f})"
                summary_row[label] = f"{row['mean_nll']:.3f}{suffix}"
            else:
                summary_row[label] = row.get("status", "error")
        table_rows.append(summary_row)
    columns = [("length", "tokens")] + [(label, label) for label in labels]
    return markdown_table(table_rows, columns)


def main() -> None:
    args = resolve_args(parse_args())
    require_hf_stack(args)
    set_seed(args.seed)

    output_dir = make_output_dir(args.output, prefix="nll-mini")
    write_json(output_dir / "environment.json", collect_environment(args))

    print(f"Writing results to {output_dir}")
    print(f"Loading model {args.model} on {args.device}")
    model, tokenizer = load_model_and_tokenizer(args)

    max_length = max(args.lengths)
    print(f"Preparing {args.num_docs} token chunk(s) from {args.dataset} up to {max_length + 1} tokens")
    docs = load_token_documents(tokenizer, args, max_length)

    thrift_ready, thrift_note = thrift_acceleration_status(args.device)
    if not thrift_ready:
        print(f"Accelerated ThriftAttention methods will be skipped if requested: {thrift_note}")

    npz_dir = output_dir / "per_token"
    npz_dir.mkdir(exist_ok=True)
    runs = method_runs(args)

    rows: list[dict[str, Any]] = []
    for run in runs:
        ok, note = configure_method(
            model,
            run["method"],
            fraction=run["fraction"],
            thrift_ready=thrift_ready,
            thrift_note=thrift_note,
        )
        print(f"\nMethod {run['label']}: {note}")
        if not ok:
            for length in args.lengths:
                rows.append(
                    {
                        "label": run["label"],
                        "method": run["method"],
                        "fraction": run["fraction"],
                        "length": length,
                        "status": "skipped",
                        "mean_nll": None,
                        "ppl": None,
                        "num_docs": 0,
                        "error": note,
                    }
                )
            continue
        for length in args.lengths:
            try:
                rows.append(run_cell(model, docs, run, length, args, npz_dir))
            except (RuntimeError, ValueError) as exc:
                message = str(exc)
                print(f"  {run['label']:<13} length={length:<7} error={message}")
                rows.append(
                    {
                        "label": run["label"],
                        "method": run["method"],
                        "fraction": run["fraction"],
                        "length": length,
                        "status": "error",
                        "mean_nll": None,
                        "ppl": None,
                        "num_docs": 0,
                        "error": message,
                    }
                )

    add_deltas(rows)
    write_jsonl(output_dir / "metrics.jsonl", rows)
    labels = [run["label"] for run in runs]
    summary_table = build_summary(rows, args.lengths, labels)

    summary = "\n".join(
        [
            "# Long-Context NLL Mini",
            "",
            "Prefill-only per-token negative log-likelihood. Lower is better; values in parentheses are delta vs fp16.",
            "",
            summary_table,
            "",
            f"Model: `{args.model}`",
            f"Dataset: `{args.dataset}`",
            f"Thrift fractions: `{', '.join(f'{fraction:g}' for fraction in args.fractions)}`",
        ]
    ).rstrip()
    (output_dir / "summary.md").write_text(summary + "\n", encoding="utf-8")

    print("\nSummary")
    print(summary_table)
    print(f"\nWrote metrics.jsonl, summary.md, environment.json, and per-token NPZ files under {output_dir}")


if __name__ == "__main__":
    main()
