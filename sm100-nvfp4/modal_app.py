from pathlib import Path
import shlex
import subprocess

import modal


LOCAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = LOCAL_DIR.parent

REMOTE_WORKDIR = "/root/sm100-nvfp4"
REMOTE_INCLUDE_DIR = "/root/csrc/include"

CUDA_VERSION = "12.8.1"
CUDA_OS = "ubuntu24.04"


image = (
    modal.Image.from_registry(
        f"nvidia/cuda:{CUDA_VERSION}-devel-{CUDA_OS}",
        add_python="3.11",
    )
    .entrypoint([])
    .add_local_dir(
        LOCAL_DIR,
        remote_path=REMOTE_WORKDIR,
        ignore=["__pycache__/**", "*.pyc"],
    )
    .add_local_dir(REPO_ROOT / "csrc" / "include", remote_path=REMOTE_INCLUDE_DIR)
)

app = modal.App("sm100-nvfp4-quantise", image=image)


def _run(cmd: list[str], cwd: str | None = None) -> None:
    print("+", shlex.join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


@app.function(gpu="B200", timeout=900)
def run_quantise_smoke(bs: int, seq_len: int, head_dim: int, arch: str) -> None:
    binary = "/tmp/nvfp4_quantise_smoke"
    workdir = Path(REMOTE_WORKDIR)

    _run(["nvidia-smi", "-L"])
    _run(["nvcc", "--version"])

    compile_cmd = [
        "nvcc",
        "-std=c++17",
        "-O3",
        "-use_fast_math",
        f"-arch={arch}",
        "-I",
        REMOTE_INCLUDE_DIR,
        str(workdir / "quantise_smoke.cu"),
        str(workdir / "quantise_nvfp4.cu"),
        "-o",
        binary,
    ]
    _run(compile_cmd, cwd=REMOTE_WORKDIR)

    _run([binary, str(bs), str(seq_len), str(head_dim)])


@app.local_entrypoint()
def main(
    bs: int = 1,
    seq_len: int = 128,
    head_dim: int = 128,
    arch: str = "sm_100a",
) -> None:
    run_quantise_smoke.remote(bs, seq_len, head_dim, arch)
