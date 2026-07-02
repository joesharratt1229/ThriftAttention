from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess


HERE = Path(__file__).resolve().parent


def run(cmd: list[str], cwd: Path | str | None = None) -> None:
    print("+", shlex.join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def nvcc_arch_flags(arch: str) -> list[str]:
    if arch.startswith("sm_"):
        return [f"-gencode=arch=compute_{arch[3:]},code={arch}"]
    return [f"-arch={arch}"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile and run the SM100 NVFP4 attention smoke test.")
    parser.add_argument("--arch", default="sm_100a")
    parser.add_argument("--include-dir", default="/root/csrc/include")
    parser.add_argument("--binary", default="/tmp/fp4_attention_sm100_smoke")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q-len", type=int, default=256)
    parser.add_argument("--kv-len", type=int, default=128)
    parser.add_argument("--num-q-heads", type=int, default=1)
    parser.add_argument("--num-kv-heads", type=int, default=1)
    args = parser.parse_args()

    compile_cmd = [
        "nvcc",
        "-std=c++17",
        "-O3",
        "-use_fast_math",
        *nvcc_arch_flags(args.arch),
        "-I",
        args.include_dir,
        str(HERE / "fp4_attention_sm100_smoke.cu"),
        str(HERE / "quantise_nvfp4.cu"),
        str(HERE / "fp4_attention_sm100.cu"),
        "-lcuda",
        "-o",
        args.binary,
    ]
    run(compile_cmd, cwd=HERE)

    run(
        [
            args.binary,
            str(args.batch),
            str(args.q_len),
            str(args.kv_len),
            str(args.num_q_heads),
            str(args.num_kv_heads),
        ]
    )


if __name__ == "__main__":
    main()
