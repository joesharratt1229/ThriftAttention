#!/usr/bin/env python3
"""Compile the SM100 attention kernel and retrieve a SASS register report.

Usage:
    modal run modal_register_report.py
    modal run modal_register_report.py --output fp4-registers.sass
    modal run modal_register_report.py --profile-build
"""
from __future__ import annotations

import gzip
import math
import re
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parent
REMOTE_SRC = "/root/sketch"

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .add_local_dir(
        ROOT,
        remote_path=REMOTE_SRC,
        ignore=["**/.torch_extensions/**", "**/__pycache__/**", "**/*.o", "**/.claude/**"],
    )
)

app = modal.App("nvfp4-sm100-register-report", image=image)


# Source ranges for the specialized role bodies in fp4_attention_sm100.cu.
# nvdisasm -gi reports both an inlined helper line and its call site; choosing
# the line which falls in one of these ranges attributes helper SASS back to
# the role that invoked it.
ROLE_RANGES = (
    ("softmax owner (w0-3)", 1117, 1235),
    ("softmax partner (w4-7)", 1238, 1296),
    ("store (w9)", 1298, 1313),
    ("load (w10)", 1315, 1337),
    ("mma (w8)", 1339, 1444),
)


def summarize_liveness(line_sass: str, live_sass: str) -> str:
    """Join separate line-info and liveness disassemblies by SASS offset."""
    samples = {name: [] for name, _, _ in ROLE_RANGES}
    by_line = {name: {} for name, _, _ in ROLE_RANGES}
    source_by_offset: dict[int, tuple[str, int]] = {}
    current_role: str | None = None
    current_source_line: int | None = None
    pending_source_lines: list[int] = []
    source_marker_pending = False
    in_attention = False

    # CUDA 13 nvdisasm explicitly refuses to combine -gi with -lrm=count.
    # Build an offset -> source-role map from the line-info-only disassembly.
    for text_line in line_sass.splitlines():
        stripped = text_line.split("//", 1)[0].strip()
        if stripped == ".text.nvfp4_sm100_attention_kernel:":
            in_attention = True
            continue
        if in_attention and stripped.startswith(".text.") and stripped.endswith(":"):
            in_attention = False
        if not in_attention:
            continue

        if "//## File" in text_line:
            if not source_marker_pending:
                pending_source_lines = []
            pending_source_lines.extend(
                int(value) for value in re.findall(r"\bline\s+(\d+)", text_line)
            )
            source_marker_pending = True
            continue

        offset_match = re.search(r"/\*([0-9a-fA-F]+)\*/", text_line)
        if offset_match is None:
            continue

        if source_marker_pending:
            current_role = None
            current_source_line = None
            for source_line in pending_source_lines:
                for role, first, last in ROLE_RANGES:
                    if first <= source_line <= last:
                        current_role = role
                        current_source_line = source_line
                        break
                if current_role is not None:
                    break
            source_marker_pending = False

        if current_role is not None and current_source_line is not None and offset_match is not None:
            source_by_offset[int(offset_match.group(1), 16)] = (current_role, current_source_line)

    in_attention = False
    kernel_max_gpr = -1
    for text_line in live_sass.splitlines():
        stripped = text_line.split("//", 1)[0].strip()
        if stripped == ".text.nvfp4_sm100_attention_kernel:":
            in_attention = True
            continue
        if in_attention and stripped.startswith(".text.") and stripped.endswith(":"):
            in_attention = False
        if not in_attention:
            continue

        offset_match = re.search(r"/\*([0-9a-fA-F]+)\*/", text_line)
        live_match = re.search(r"//\s*\|\s*(\d+)\b", text_line)
        instruction = text_line.split("//", 1)[0]
        gprs = [int(value) for value in re.findall(r"(?<!U)\bR(\d+)\b", instruction)]
        if gprs:
            kernel_max_gpr = max(kernel_max_gpr, max(gprs))
        if offset_match is None or live_match is None:
            continue
        source = source_by_offset.get(int(offset_match.group(1), 16))
        if source is None:
            continue

        current_role, current_source_line = source
        live = int(live_match.group(1))
        samples[current_role].append(live)
        by_line[current_role].setdefault(current_source_line, []).append(live)

    rows = []
    for role, first, last in ROLE_RANGES:
        values = samples[role]
        if not values:
            rows.append((role, first, last, 0, None, None, []))
            continue
        ordered = sorted(values)
        p90 = ordered[math.ceil(0.90 * len(ordered)) - 1]
        hottest = sorted(
            ((max(line_values), line, len(line_values))
             for line, line_values in by_line[role].items()),
            reverse=True,
        )[:5]
        rows.append((
            role,
            first,
            last,
            len(values),
            max(values),
            p90,
            hottest,
        ))

    output = [
        "=== static register liveness by specialized role ===",
        "",
        f"{'role':<27} {'SASS inst':>9} {'peak live':>10} {'p90 live':>9}",
    ]
    for role, _first, _last, count, peak, p90, _hottest in rows:
        output.append(
            f"{role:<27} {count:>9} "
            f"{str(peak) if peak is not None else 'n/a':>10} "
            f"{str(p90) if p90 is not None else 'n/a':>9}"
        )

    output.extend([
        "",
        "peak/p90 are static occupied-GPR counts, not runtime-weighted averages.",
        (f"Highest explicit attention-kernel GPR operand: R{kernel_max_gpr} "
         f"({kernel_max_gpr + 1} registers)."
         if kernel_max_gpr >= 0 else
         "No attention-kernel GPR operands found."),
    ])
    for role, first, last, _count, _peak, _p90, hottest in rows:
        output.extend(["", f"{role} (CUDA lines {first}-{last}), hottest lines:"])
        if not hottest:
            output.append("  no line-correlated SASS found")
            continue
        for peak, source_line, instruction_count in hottest:
            output.append(
                f"  line {source_line:<4}  peak live {peak:>3}  "
                f"({instruction_count} SASS instructions)"
            )
    return "\n".join(output)


@app.function(timeout=900)
def build_report(profile_build: bool = False) -> tuple[str, bytes]:
    import subprocess

    work = Path("/tmp/fp4-register-report")
    work.mkdir(parents=True, exist_ok=True)
    cubin = work / "fp4.cubin"

    nvcc_cmd = [
        "/usr/local/cuda/bin/nvcc",
        str(Path(REMOTE_SRC) / "fp4_attention_sm100.cu"),
        "-cubin",
        "-o", str(cubin),
        "-O3",
        "-std=c++17",
        "-gencode=arch=compute_100a,code=sm_100a",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--relocatable-device-code=false",
        "-lineinfo",
        "-Xptxas=-v,-warn-spills"
    ]
    if profile_build:
        nvcc_cmd.append("-DFA4_PROF=1")

    def run(command: list[str]) -> str:
        result = subprocess.run(
            command,
            cwd=REMOTE_SRC,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"command failed with exit code {result.returncode}:\n"
                f"{' '.join(command)}\n\n{result.stdout}"
            )
        return result.stdout

    compile_log = run(nvcc_cmd)
    resource_log = run([
        "/usr/local/cuda/bin/cuobjdump", "-res-usage", str(cubin),
    ])
    line_sass = run([
        "/usr/local/cuda/bin/nvdisasm",
        "-gi",
        str(cubin),
    ])
    live_sass = run([
        "/usr/local/cuda/bin/nvdisasm",
        "-lrm=count",
        str(cubin),
    ])
    liveness_summary = summarize_liveness(line_sass, live_sass)

    summary = (
        "=== nvcc / ptxas ===\n"
        f"{compile_log.rstrip()}\n\n"
        "=== cuobjdump -res-usage ===\n"
        f"{resource_log.rstrip()}\n\n"
        f"{liveness_summary}\n"
    )
    combined_sass = (
        "// LINE-CORRELATED DISASSEMBLY\n"
        f"{line_sass}\n"
        "// REGISTER-LIVENESS DISASSEMBLY\n"
        f"{live_sass}"
    )
    return summary, gzip.compress(combined_sass.encode("utf-8"), compresslevel=6)


@app.local_entrypoint()
def main(output: str = "fp4-registers.sass", profile_build: bool = False) -> None:
    summary, compressed_sass = build_report.remote(profile_build=profile_build)
    output_path = Path(output).expanduser().resolve()
    output_path.write_bytes(gzip.decompress(compressed_sass))
    print(summary)
    print(f"Full line-correlated SASS register report written to {output_path}")
