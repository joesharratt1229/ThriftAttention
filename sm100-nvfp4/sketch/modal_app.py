from pathlib import Path
import shlex
import subprocess

import modal


LOCAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LOCAL_DIR.parent.parent

REMOTE_WORKDIR = "/root/sm100-nvfp4"
REMOTE_INCLUDE_DIR = "/root/csrc/include"

CUDA_VERSION = "13.0.2"
CUDA_OS = "ubuntu24.04"
CUDA_FLAVOR = "cudnn-devel"


image = (
    modal.Image.from_registry(
        f"nvidia/cuda:{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}",
        add_python="3.11",
    )
    .entrypoint([])
    .add_local_dir(
        LOCAL_DIR,
        remote_path=REMOTE_WORKDIR,
        ignore=["__pycache__/**", "*.pyc"],
    )
    .add_local_dir(PROJECT_ROOT / "csrc" / "include", remote_path=REMOTE_INCLUDE_DIR)
)

app = modal.App("sm100-nvfp4-attention", image=image)


def _run(cmd: list[str], cwd: str | None = None) -> None:
    print("+", shlex.join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _nvcc_arch_flags(arch: str) -> list[str]:
    if arch.startswith("sm_"):
        return [f"-gencode=arch=compute_{arch[3:]},code={arch}"]
    return [f"-arch={arch}"]


@app.function(gpu="B200", timeout=1800)
def run_quantise_then_attention(
    bs: int,
    q_len: int,
    kv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    arch: str,
) -> None:
    binary = "/tmp/nvfp4_quantise_smoke"
    workdir = Path(REMOTE_WORKDIR)

    _run(["nvidia-smi", "-L"])
    _run(["nvcc", "--version"])

    compile_cmd = [
        "nvcc",
        "-std=c++17",
        "-O3",
        "-use_fast_math",
        *_nvcc_arch_flags(arch),
        "-I",
        REMOTE_INCLUDE_DIR,
        str(workdir / "quantise_smoke.cu"),
        str(workdir / "quantise_nvfp4.cu"),
        "-o",
        binary,
    ]
    _run(compile_cmd, cwd=REMOTE_WORKDIR)

    _run([binary, str(bs), str(q_len), "128"])
    _run(
        [
            "python3",
            str(workdir / "fp4_attention_sm100.py"),
            "--arch",
            arch,
            "--include-dir",
            REMOTE_INCLUDE_DIR,
            "--binary",
            "/tmp/fp4_attention_sm100_smoke",
            "--batch",
            str(bs),
            "--q-len",
            str(q_len),
            "--kv-len",
            str(kv_len),
            "--num-q-heads",
            str(num_q_heads),
            "--num-kv-heads",
            str(num_kv_heads),
        ],
        cwd=REMOTE_WORKDIR,
    )


@app.local_entrypoint()
def main(
    bs: int = 1,
    q_len: int = 256,
    kv_len: int = 128,
    num_q_heads: int = 1,
    num_kv_heads: int = 1,
    arch: str = "sm_100a",
) -> None:
    run_quantise_then_attention.remote(bs, q_len, kv_len, num_q_heads, num_kv_heads, arch)
