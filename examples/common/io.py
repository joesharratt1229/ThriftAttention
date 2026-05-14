from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


def make_output_dir(root: str | Path, *, prefix: str = "run") -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = Path(root) / f"{timestamp}-{prefix}"
    suffix = 1
    candidate = path
    while candidate.exists():
        suffix += 1
        candidate = Path(f"{path}-{suffix}")
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def markdown_table(rows: Sequence[dict[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    headers = [label for _, label in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = [_format_cell(row.get(key, "")) for key, _ in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_summary_md(
    path: str | Path,
    *,
    title: str,
    rows: Sequence[dict[str, Any]],
    columns: Sequence[tuple[str, str]],
    notes: Sequence[str] | None = None,
) -> None:
    sections = [f"# {title}", "", markdown_table(rows, columns)]
    if notes:
        sections.extend(["", *notes])
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")


def _format_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value == float("inf"):
            return "inf"
        return f"{value:.4g}"
    return str(value).replace("\n", " ")
