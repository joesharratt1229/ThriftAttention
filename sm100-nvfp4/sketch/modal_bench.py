#!/usr/bin/env python3
"""Run bench_attention.py on a Modal B200.

Usage (from this directory, needs `pip install modal` + `modal setup` locally):

    modal run modal_bench.py
    modal run modal_bench.py --seqlens "4096,8192,16384" --iters 200

The sketch directory is shipped to the container at run time (not baked into
the image), so editing the .cu locally and re-running picks up the change
without an image rebuild.  The torch extension is compiled on first run
inside the container (~2-3 min); subsequent runs in a warm container reuse
the ninja cache.

Image notes:
  - nvidia/cuda 13.0 devel + torch cu130 wheels matches the box this kernel
    was developed on (B200, driver 580, cuda 13.0).  If the cu130 wheel index
    is unavailable, drop both to 12.8 (nvidia/cuda:12.8.1-devel-ubuntu24.04 +
    --index-url .../whl/cu128); sm_100a compiles fine on 12.8.
  - flash-attn is installed with FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE:
    flash_attn.cute (the FA4 baseline) is pure Python over the CuTe DSL, so
    the heavyweight C++/CUDA build is unnecessary.  If either FA4 dep fails
    to install, bench_attention.py degrades gracefully and still runs the
    base-vs-INT_ENCODE comparison (which is the decision-relevant part).
"""
from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
REMOTE_SRC = "/root/sketch"
REMOTE_WORK = "/root/work"

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install("ninja", "numpy")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    # FA4 baseline (optional -- see module docstring).  flash-attn's sdist
    # setup.py imports psutil/packaging/einops even with the CUDA build
    # skipped, and --no-build-isolation means they must be preinstalled
    # (first Modal run failed here: ModuleNotFoundError -> fa4 column skipped).
    .pip_install("psutil", "packaging", "einops")
    .run_commands(
        "pip install nvidia-cutlass-dsl || true",
        "FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install --no-build-isolation flash-attn || true",
    )
    .add_local_dir(
        ROOT,
        remote_path=REMOTE_SRC,
        ignore=["**/.torch_extensions/**", "**/__pycache__/**", "**/*.o", "**/.claude/**"],
    )
)

app = modal.App("nvfp4-sm100-bench", image=image)


@app.function(gpu="B200", timeout=3600)
def bench(batch: int = 1,
          heads: int = 16,
          seqlens: str = "1024,2048,4096,8192,16384",
          warmup: int = 10,
          iters: int = 100,
          cmd: str = "") -> str:
    import os
    import shutil
    import subprocess
    import sys

    # The add_local_dir mount is read-only; the torch extension build writes
    # .torch_extensions/ next to the sources, so work from a writable copy.
    shutil.copytree(REMOTE_SRC, REMOTE_WORK, dirs_exist_ok=True)
    # Ops note from RESTRUCTURE_PLAN.txt: a killed build leaves a ninja lock
    # that blocks every later build forever.  Fresh copy => no lock, but be
    # explicit in case the container is reused.
    subprocess.run(["find", REMOTE_WORK, "-name", "lock", "-path", "*.torch_extensions*",
                    "-delete"], check=False)

    env = dict(os.environ)
    env.setdefault("CUDA_HOME", "/usr/local/cuda")
    # Let ld resolve -lcuda from the toolkit stubs at build time; at run time
    # the driver's libcuda.so.1 is injected by the container runtime.
    env["LIBRARY_PATH"] = "/usr/local/cuda/lib64/stubs:" + env.get("LIBRARY_PATH", "")

    if cmd:
        # Arbitrary sketch script, e.g.
        #   --cmd "profile_breakdown.py --ablate --seqlens 8192"
        argv = [sys.executable, "-u", *cmd.split()]
    else:
        argv = [sys.executable, "-u", "bench_attention.py",
                "--batch", str(batch), "--heads", str(heads),
                "--warmup", str(warmup), "--iters", str(iters),
                "--seqlens", *seqlens.replace(",", " ").split()]
    proc = subprocess.Popen(argv, cwd=REMOTE_WORK, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"bench_attention.py exited with rc={rc}")
    return "".join(lines)


@app.local_entrypoint()
def main(batch: int = 1,
         heads: int = 16,
         seqlens: str = "1024,2048,4096,8192,16384",
         warmup: int = 10,
         iters: int = 100,
         cmd: str = "") -> None:
    bench.remote(batch=batch, heads=heads, seqlens=seqlens,
                 warmup=warmup, iters=iters, cmd=cmd)
