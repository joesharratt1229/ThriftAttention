#!/usr/bin/env python3
"""Analyse run_nll_per_token.py runs that included the `probe` method.

Answers, from the saved probe/* arrays and the per-token NLL:
  1. Is quantisation error concentrated in few blocks? (cumulative curves:
     oracle ordering vs the block_mean heuristic ranking vs the uniform line)
  2. How much of that error do the blocks ThriftAttention selects hold?
  3. Does captured error predict NLL recovery
     (nll_fp4 - nll_thrift) / (nll_fp4 - nll_fp16)?

Question 3 is a regression, so it needs points. Pass several run directories --
one per context length -- and every (length, budget) cell becomes one point,
plus per-document points inside each cell:

  python analyze_block_error.py results/nll_per_token/*-4096 ... *-131072

Writes probe_summary.md and curves.csv into each run directory, and
calibration.csv + calibration.md into the first one when several are given.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from block_error_probe import GRID


def load_run(run_dir: Path) -> tuple[dict, list[dict], dict]:
    arrays = dict(np.load(run_dir / "per_token_nll.npz"))
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    environment = json.loads((run_dir / "environment.json").read_text())
    return arrays, rows, environment


def probe_docs(arrays: dict) -> list[int]:
    return sorted(
        int(key.rsplit("doc", 1)[1]) for key in arrays if key.startswith("probe/err_total/doc")
    )


def pooled(arrays: dict, name: str, docs: list[int]) -> np.ndarray:
    """Sum an accumulator over documents -> [layers, heads, ...]."""
    return np.sum([arrays[f"probe/{name}/doc{i}"] for i in docs], axis=0)


def doc_mean_nll(arrays: dict, label: str, doc: int) -> float:
    return float(arrays[f"nll/{label}/doc{doc}"].mean())


def recovery(arrays: dict, label: str, docs: list[int]) -> float:
    """(nll_fp4 - nll_thrift) / (nll_fp4 - nll_fp16), pooled over documents."""
    fp16 = np.mean([doc_mean_nll(arrays, "fp16", i) for i in docs])
    fp4 = np.mean([doc_mean_nll(arrays, "fp4", i) for i in docs])
    thrift = np.mean([doc_mean_nll(arrays, label, i) for i in docs])
    return float((fp4 - thrift) / (fp4 - fp16)) if fp4 != fp16 else float("nan")


def fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Least-squares line plus Pearson r; nan when there is nothing to fit."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2 or np.ptp(x) == 0:
        return {"n": int(x.size), "slope": float("nan"), "intercept": float("nan"), "r": float("nan")}
    slope, intercept = np.polyfit(x, y, 1)
    return {
        "n": int(x.size),
        "slope": float(slope),
        "intercept": float(intercept),
        "r": float(np.corrcoef(x, y)[0, 1]),
    }


def analyse_run(run_dir: Path, method: str) -> list[dict]:
    """Write the per-run report; return one record per (budget) cell."""
    arrays, rows, environment = load_run(run_dir)
    docs = probe_docs(arrays)
    if not docs:
        raise SystemExit(f"{run_dir} has no probe arrays; run with --methods ...,probe")

    probe_row = next(row for row in rows if row["method"] == "probe")
    budgets = [entry["budget"] for entry in probe_row["probe"]]
    length = environment["length"]

    err_total = pooled(arrays, "err_total", docs)          # [layers, heads]
    mass_total = pooled(arrays, "mass_total", docs)
    err_captured = pooled(arrays, "err_captured", docs)    # [layers, heads, budgets]
    err_resid = pooled(arrays, "err_resid", docs)
    mass_captured = pooled(arrays, "mass_captured", docs)
    has_resid = err_resid.sum() > 0

    records = []
    for bi, budget in enumerate(budgets):
        label = f"{method}_{budget * 100:g}pct"
        if f"nll/{label}/doc{docs[0]}" not in arrays:
            continue
        record = {
            "length": length,
            "budget": budget,
            "err_capture": float(err_captured[..., bi].sum() / err_total.sum()),
            "mass_capture": float(mass_captured[..., bi].sum() / mass_total.sum()),
            "recovery": recovery(arrays, label, docs),
            "per_doc": [
                (
                    float(
                        arrays[f"probe/err_captured/doc{i}"][..., bi].sum()
                        / arrays[f"probe/err_total/doc{i}"].sum()
                    ),
                    float(
                        (doc_mean_nll(arrays, "fp4", i) - doc_mean_nll(arrays, label, i))
                        / (doc_mean_nll(arrays, "fp4", i) - doc_mean_nll(arrays, "fp16", i))
                    ),
                )
                for i in docs
            ],
        }
        if has_resid:
            record["err_removed"] = float(1 - err_resid[..., bi].sum() / err_total.sum())
        records.append(record)

    header = "| budget | err capture | mass capture |" + (" err removed |" if has_resid else "") + " NLL recovery |"
    lines = [
        f"# Block quantisation error vs NLL recovery ({length} tokens)",
        "",
        f"{len(docs)} docs; total P-error (summed over layers/heads/docs) = {err_total.sum():.4g}",
        "",
        header,
        "|---|---|---|" + ("---|" if has_resid else "") + "---|",
    ]
    for record in records:
        removed = f" {record['err_removed']:.3f} |" if has_resid else ""
        lines.append(
            f"| {record['budget'] * 100:g}% | {record['err_capture']:.3f} "
            f"| {record['mass_capture']:.3f} |{removed} {record['recovery']:.3f} |"
        )

    # concentration curves, pooled over layers/heads/docs
    curves = {
        name: pooled(arrays, name, docs).sum(axis=(0, 1))
        for name in ("curve_err_oracle", "curve_err_ranked", "curve_mass_oracle")
    }
    curve_lines = ["fraction,err_oracle,err_ranked,mass_oracle,uniform"]
    for gi, fraction in enumerate(GRID):
        curve_lines.append(
            f"{fraction},{curves['curve_err_oracle'][gi] / err_total.sum():.6f}"
            f",{curves['curve_err_ranked'][gi] / err_total.sum():.6f}"
            f",{curves['curve_mass_oracle'][gi] / mass_total.sum():.6f},{fraction}"
        )
    (run_dir / "curves.csv").write_text("\n".join(curve_lines) + "\n")

    lines += ["", "## Per-layer error capture", "",
              "| layer | " + " | ".join(f"{r['budget'] * 100:g}%" for r in records) + " |",
              "|---|" + "---|" * len(records)]
    for layer in range(err_total.shape[0]):
        if err_total[layer].sum() == 0:
            continue  # skipped by --probe-layer-stride
        cells = " | ".join(
            f"{err_captured[layer, :, bi].sum() / err_total[layer].sum():.3f}" for bi in range(len(records))
        )
        lines.append(f"| {layer} | {cells} |")

    lines += ["", "## Concentration curves (pooled cumulative fraction)", "",
              "| fraction of blocks | err oracle | err thrift-ranked | mass oracle |", "|---|---|---|---|"]
    for gi, fraction in enumerate(GRID):
        lines.append(
            f"| {fraction:g} | {curves['curve_err_oracle'][gi] / err_total.sum():.3f} "
            f"| {curves['curve_err_ranked'][gi] / err_total.sum():.3f} "
            f"| {curves['curve_mass_oracle'][gi] / mass_total.sum():.3f} |"
        )

    report = "\n".join(lines) + "\n"
    (run_dir / "probe_summary.md").write_text(report)
    print(report)
    return records


def calibration(records: list[dict], out_dir: Path) -> str:
    """Regress NLL recovery on captured error across cells and across documents."""
    cells = np.array([[r["err_capture"], r["recovery"]] for r in records])
    cell_fit = fit(cells[:, 0], cells[:, 1])
    per_doc = np.array([point for r in records for point in r["per_doc"]])
    doc_fit = fit(per_doc[:, 0], per_doc[:, 1])

    lines = [
        "# Does captured quantisation error predict NLL recovery?",
        "",
        "| level | n | Pearson r | slope | intercept |",
        "|---|---|---|---|---|",
        f"| (length, budget) cells | {cell_fit['n']} | {cell_fit['r']:.3f} | {cell_fit['slope']:.3f} "
        f"| {cell_fit['intercept']:.3f} |",
        f"| individual documents | {doc_fit['n']} | {doc_fit['r']:.3f} | {doc_fit['slope']:.3f} "
        f"| {doc_fit['intercept']:.3f} |",
        "",
        "A slope near 1 with a small intercept would mean captured error translates",
        "one-for-one into recovered quality; a high r with slope far from 1 still",
        "means error predicts recovery, just on a rescaled axis.",
        "",
        "| length | budget | err capture | mass capture | NLL recovery |",
        "|---|---|---|---|---|",
    ]
    for record in sorted(records, key=lambda r: (r["length"], r["budget"])):
        lines.append(
            f"| {record['length']} | {record['budget'] * 100:g}% | {record['err_capture']:.3f} "
            f"| {record['mass_capture']:.3f} | {record['recovery']:.3f} |"
        )

    csv = ["length,budget,err_capture,mass_capture,recovery"]
    csv += [
        f"{r['length']},{r['budget']},{r['err_capture']:.6f},{r['mass_capture']:.6f},{r['recovery']:.6f}"
        for r in sorted(records, key=lambda r: (r["length"], r["budget"]))
    ]
    (out_dir / "calibration.csv").write_text("\n".join(csv) + "\n")
    report = "\n".join(lines) + "\n"
    (out_dir / "calibration.md").write_text(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+", help="one run directory per context length")
    parser.add_argument("--method", default="block_mean", help="thrift method the probe's selection matches")
    args = parser.parse_args()

    records = [record for run_dir in args.run_dirs for record in analyse_run(run_dir, args.method)]
    if not records:
        raise SystemExit(
            f"no probe budget matched a {args.method} NLL run; "
            f"the sweep must run --methods {args.method},fp16,fp4,probe on the same budgets"
        )
    report = calibration(records, args.run_dirs[0])
    print(report)
    print(f"Wrote {args.run_dirs[0] / 'calibration.md'} and calibration.csv")


if __name__ == "__main__":
    main()
