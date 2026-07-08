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
    .pip_install("psutil", "packaging", "einops")
    .run_commands(
        "pip install nvidia-cutlass-dsl || true",
        "FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install --no-build-isolation flash-attn || true",
    )
    .add_local_dir(
        ROOT,
        remote_path=REMOTE_SRC,
        ignore=["**/.torch_extensions/**", "**/__pycache__/**", "**/*.o", "**/.git*"],
    )
)

app = modal.App("nvfp4-sm100-bench", image=image)


@app.function(gpu="B200", timeout=3600)
def bench(batch: int = 1,
          heads: int = 16,
          seqlens: str = "1024,2048,4096,8192,16384",
          warmup: int = 10,
          iters: int = 100) -> str:
    import os
    import shutil
    import subprocess
    import sys

    shutil.copytree(REMOTE_SRC, REMOTE_WORK, dirs_exist_ok=True)
    subprocess.run(["find", REMOTE_WORK, "-name", "lock", "-path", "*.torch_extensions*",
                    "-delete"], check=False)

    env = dict(os.environ)
    env.setdefault("CUDA_HOME", "/usr/local/cuda")
    env["LIBRARY_PATH"] = "/usr/local/cuda/lib64/stubs:" + env.get("LIBRARY_PATH", "")

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
         iters: int = 100) -> None:
    bench.remote(batch=batch, heads=heads, seqlens=seqlens,
                 warmup=warmup, iters=iters)
