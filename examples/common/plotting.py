from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


def plot_quality_vs_length(rows: Sequence[dict[str, Any]], path: str | Path, *, metric: str = "nll") -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required to write plots. Install with `pip install -e '.[plots]'` "
            "or use the example requirements.txt."
        ) from exc

    grouped: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        if row.get("status") != "ok" or row.get(metric) is None:
            continue
        grouped.setdefault(str(row["method"]), []).append((int(row["length"]), float(row[metric])))

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if grouped:
        for method, points in sorted(grouped.items()):
            points = sorted(points)
            ax.plot([x for x, _ in points], [y for _, y in points], marker="o", label=method)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No successful rows to plot", ha="center", va="center", transform=ax.transAxes)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel(metric.upper() if metric != "nll" else "Next-token NLL")
    ax.set_title("Quality vs. context length")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
