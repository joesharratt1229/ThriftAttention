#!/usr/bin/env python3
"""Profile nvfp4_sm100_attention_kernel with Nsight Compute on Modal.

Usage:

    modal run modal_ncu_profile.py
    modal run modal_ncu_profile.py --seqlen 8192 --warmup 10
    modal run modal_ncu_profile.py --query-metrics --output fp4-ncu.txt

The first matching attention launch constructs the pre-quantised inputs.
After that, ``warmup`` matching launches warm the kernel.  Nsight Compute
therefore skips ``warmup + 1`` launches and profiles the next launch.

This script can invoke ncu correctly, but Modal's host must expose and permit
access to NVIDIA performance counters.  If it does not, ncu reports either
ERR_NVGPUCTRPERM or a counter-library ``LibraryNotLoaded`` error.  Neither
condition can be fixed from this container.
"""
from __future__ import annotations

from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parent
REMOTE_SRC = "/root/sketch"
REMOTE_WORK = "/root/ncu-work"

# Keep this image deliberately smaller than modal_bench.py's image.  The
# FlashAttention comparison is not needed here; bench_attention.py degrades
# gracefully when it is absent and then launches only our attention kernel.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12"
    )
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install("ninja", "numpy")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .add_local_dir(
        ROOT,
        remote_path=REMOTE_SRC,
        ignore=[
            "**/.torch_extensions/**",
            "**/__pycache__/**",
            "**/*.o",
            "**/.claude/**",
        ],
    )
)

app = modal.App("nvfp4-sm100-ncu-profile", image=image)


@app.function(gpu="B200", timeout=3600)
def profile(
    batch: int = 1,
    heads: int = 16,
    seqlen: int = 8192,
    warmup: int = 10,
    query_metrics: bool = False,
) -> str:
    import os
    import re
    import shlex
    import shutil
    import subprocess
    import sys

    if batch < 1 or heads < 1 or seqlen < 1 or warmup < 0:
        raise ValueError("batch, heads and seqlen must be positive; warmup must be >= 0")

    shutil.copytree(REMOTE_SRC, REMOTE_WORK, dirs_exist_ok=True)
    subprocess.run(
        [
            "find",
            REMOTE_WORK,
            "-name",
            "lock",
            "-path",
            "*.torch_extensions*",
            "-delete",
        ],
        check=False,
    )

    env = dict(os.environ)
    env.setdefault("CUDA_HOME", "/usr/local/cuda")
    env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")
    env["LIBRARY_PATH"] = (
        "/usr/local/cuda/lib64/stubs:" + env.get("LIBRARY_PATH", "")
    )

    ncu = shutil.which("ncu", path=env["PATH"])
    if ncu is None:
        candidates = sorted(Path("/opt/nvidia/nsight-compute").glob("*/ncu"))
        if candidates:
            ncu = str(candidates[-1])
    if ncu is None:
        raise RuntimeError(
            "ncu was not found in the CUDA devel image. Checked PATH and "
            "/opt/nvidia/nsight-compute/*/ncu."
        )

    def run_streaming(command: list[str], heading: str) -> tuple[int, str]:
        print(f"\n=== {heading} ===", flush=True)
        print(shlex.join(command), flush=True)
        proc = subprocess.Popen(
            command,
            cwd=REMOTE_WORK,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        lines: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            lines.append(line)
        return proc.wait(), "".join(lines)

    version = subprocess.run(
        [ncu, "--version"],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print("=== Nsight Compute ===", flush=True)
    print(version.stdout.rstrip(), flush=True)

    if query_metrics:
        query = subprocess.run(
            [
                ncu,
                "--devices",
                "0",
                "--query-metrics",
                "--query-metrics-mode",
                "all",
            ],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        print("\n=== relevant B200 metric names ===", flush=True)
        metric_pattern = re.compile(
            r"sm(?:sp)?__.*(?:pipe_(?:fma|alu|xu|tmem)|"
            r"warps_eligible|issue_active|warp_issue_stalled)",
            re.IGNORECASE,
        )
        matches = [line for line in query.stdout.splitlines() if metric_pattern.search(line)]
        print("\n".join(matches) if matches else query.stdout.rstrip(), flush=True)
        if query.returncode != 0:
            raise RuntimeError(
                f"ncu metric query failed with exit code {query.returncode}"
            )

    # Compile before starting ncu so compilation and extension-cache activity
    # are not part of the profiling target process.
    build_command = [
        sys.executable,
        "-u",
        "-c",
        (
            "from run_fp4_attention import build_extension; "
            "build_extension(verbose=True)"
        ),
    ]
    build_rc, _ = run_streaming(build_command, "building production extension")
    if build_rc != 0:
        raise RuntimeError(f"extension build failed with exit code {build_rc}")

    launch_skip = warmup + 1
    ncu_command = [
        ncu,
        "--kernel-name-base",
        "function",
        "--kernel-name",
        "nvfp4_sm100_attention_kernel",
        "--launch-skip",
        str(launch_skip),
        "--launch-count",
        "1",
        "--section",
        "ComputeWorkloadAnalysis",
        "--section",
        "SchedulerStats",
        "--section",
        "WarpStateStats",
        "--section",
        "InstructionStats",
        "--print-details",
        "all",
        "--print-metric-name",
        "label-name",
        "--page",
        "details",
        sys.executable,
        "-u",
        "bench_attention.py",
        "--batch",
        str(batch),
        "--heads",
        str(heads),
        "--seqlens",
        str(seqlen),
        "--warmup",
        str(warmup),
        "--iters",
        "1",
    ]
    profile_rc, output = run_streaming(
        ncu_command,
        f"profiling attention (skip {launch_skip}, collect 1)",
    )

    if "ERR_NVGPUCTRPERM" in output:
        raise RuntimeError(
            "Modal's host driver denied access to NVIDIA GPU performance "
            "counters (ERR_NVGPUCTRPERM). The ncu command reached the B200, "
            "but this permission must be enabled by Modal; it cannot be "
            "granted from the image or Python function."
        )
    if "LibraryNotLoaded" in output:
        raise RuntimeError(
            "Modal's GPU runtime did not expose a driver interface compatible "
            "with Nsight Compute's counter measurement library "
            "(LibraryNotLoaded). ncu attached to the target successfully, but "
            "failed before measuring the kernel. This is a Modal host/runtime "
            "configuration issue, not a kernel, launch-skip, or section issue; "
            "it cannot be fixed by changing this container's library path."
        )
    if profile_rc != 0:
        raise RuntimeError(f"ncu exited with code {profile_rc}")
    return output


@app.local_entrypoint()
def main(
    batch: int = 1,
    heads: int = 16,
    seqlen: int = 8192,
    warmup: int = 10,
    query_metrics: bool = False,
    output: str = "",
) -> None:
    result = profile.remote(
        batch=batch,
        heads=heads,
        seqlen=seqlen,
        warmup=warmup,
        query_metrics=query_metrics,
    )
    if output:
        output_path = Path(output).expanduser().resolve()
        output_path.write_text(result)
        print(f"Nsight Compute text report written to {output_path}")
