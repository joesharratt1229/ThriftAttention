#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
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
    write_json,
    write_jsonl,
)
from common.cli import pick  # noqa: E402


DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_TASKS = "json_kv,retrieval,long_qa"
DEFAULT_METHODS = "fp16,fp4,thrift"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a tiny HELMET-style config for representative long-context generation tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", default="quick", choices=["quick"])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--lengths", default=None)
    parser.add_argument("--tasks", default=None, help="Comma-separated task names, e.g. json_kv,retrieval,long_qa.")
    parser.add_argument("--methods", default=None)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--helmet-root", type=Path, default=None, help="Optional path to an official HELMET checkout.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    config_path = args.config or Path(__file__).parent / "configs" / "helmet_quick.yaml"
    config = load_config(config_path)
    args.config_path = str(config_path)
    args.model = pick("model", args.model, config, DEFAULT_MODEL, str)
    args.lengths = pick("lengths", args.lengths, config, [4096, 8192], parse_int_list)
    args.tasks = pick("tasks", args.tasks, config, DEFAULT_TASKS, parse_str_list)
    args.methods = pick("methods", args.methods, config, DEFAULT_METHODS, parse_str_list)
    args.fraction = pick("fraction", args.fraction, config, 0.05, float)
    args.num_examples = pick("num_examples", args.num_examples, config, 2, int)
    args.output = Path(pick("output", args.output, config, Path("results/long_context_quality"), Path))
    args.seed = pick("seed", args.seed, config, 1234, int)
    args.device = pick("device", args.device, config, "cuda", str)
    return args


def build_helmet_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "name": "thriftattention_helmet_mini",
        "description": "Tiny HELMET-style smoke config; not a full HELMET evaluation.",
        "model": args.model,
        "device": args.device,
        "seed": args.seed,
        "context_lengths": args.lengths,
        "num_examples_per_task": args.num_examples,
        "tasks": [
            {
                "name": task,
                "category": _task_category(task),
                "context_lengths": args.lengths,
                "num_examples": args.num_examples,
            }
            for task in args.tasks
        ],
        "methods": [
            {
                "name": method,
                "attention": "transformers_fp16" if method in {"fp16", "flash", "fp16_flash"} else method,
                "thrift_fraction": args.fraction if method == "thrift" else None,
            }
            for method in args.methods
        ],
    }


def _task_category(task: str) -> str:
    if task == "json_kv":
        return "structured_retrieval"
    if "retrieval" in task or "rag" in task:
        return "retrieval_or_rag"
    if "qa" in task:
        return "long_qa"
    return "custom"


def write_yaml_like(path: Path, data: Any) -> None:
    path.write_text(_format_yaml(data), encoding="utf-8")


def _format_yaml(data: Any, indent: int = 0) -> str:
    prefix = " " * indent
    if isinstance(data, dict):
        lines: list[str] = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(_format_yaml(value, indent + 2).rstrip())
            else:
                lines.append(f"{prefix}{key}: {_format_scalar(value)}")
        return "\n".join(lines) + "\n"
    if isinstance(data, list):
        lines = []
        for value in data:
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}-")
                lines.append(_format_yaml(value, indent + 2).rstrip())
            else:
                lines.append(f"{prefix}- {_format_scalar(value)}")
        return "\n".join(lines) + "\n"
    return f"{prefix}{_format_scalar(data)}\n"


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if any(ch in text for ch in ":#[]{}") or text.strip() != text:
        return repr(text)
    return text


def build_commands(args: argparse.Namespace, config_path: Path) -> str:
    helmet_root = args.helmet_root or Path("/path/to/HELMET")
    return "\n".join(
        [
            "# Edit HELMET_ROOT if needed, then run from a shell.",
            f"HELMET_ROOT={helmet_root}",
            f"CONFIG={config_path.resolve()}",
            'cd "$HELMET_ROOT"',
            'python run.py --config "$CONFIG"',
            "",
            "# If your HELMET checkout uses a different entrypoint, keep the generated config",
            "# and pass it to that entrypoint with the same model, task, and length settings.",
            "",
        ]
    )


def main() -> None:
    args = resolve_args(parse_args())
    output_dir = make_output_dir(args.output, prefix="helmet-mini")
    write_json(output_dir / "environment.json", collect_environment(args))
    config = build_helmet_config(args)
    config_path = output_dir / "helmet_mini_config.yaml"
    write_yaml_like(config_path, config)
    commands = build_commands(args, config_path)
    (output_dir / "run_commands.sh").write_text(commands, encoding="utf-8")

    rows = [
        {
            "method": method,
            "task": task,
            "length": length,
            "status": "config_generated",
            "num_examples": args.num_examples,
            "fraction": args.fraction if method == "thrift" else None,
        }
        for method in args.methods
        for task in args.tasks
        for length in args.lengths
    ]
    write_jsonl(output_dir / "metrics.jsonl", rows)

    summary_rows = [
        {"task": task, "category": _task_category(task), "lengths": ", ".join(str(length) for length in args.lengths)}
        for task in args.tasks
    ]
    summary = "\n".join(
        [
            "# HELMET Mini Config",
            "",
            "This script generated a tiny HELMET-style config. It does not run a full HELMET evaluation by default.",
            "",
            markdown_table(summary_rows, [("task", "task"), ("category", "category"), ("lengths", "lengths")]),
            "",
            f"Model: `{args.model}`",
            f"Methods: `{', '.join(args.methods)}`",
            "",
            "Command file: `run_commands.sh`",
        ]
    )
    (output_dir / "summary.md").write_text(summary + "\n", encoding="utf-8")
    print(f"Wrote HELMET mini config, command file, metrics.jsonl, summary.md, and environment.json under {output_dir}")
    print(commands)


if __name__ == "__main__":
    main()
