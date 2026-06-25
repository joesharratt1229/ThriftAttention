from pathlib import Path
from dataclasses import dataclass
import os
import re
import subprocess

from setuptools import find_packages, setup

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    raise RuntimeError(
        "ThriftAttention builds against the PyTorch wheel installed in the "
        "target environment. Install torch>=2.8.0 with CUDA>=12.8 first, "
        "then run `python -m pip install -e . --no-build-isolation`."
    ) from exc


ROOT = Path(__file__).parent
MIN_TORCH = (2, 8, 0)


@dataclass(frozen=True)
class CudaArchitecture:
    torch_arch: str
    min_cuda: tuple[int, int]
    sources: tuple[str, ...]
    nvcc_flags: tuple[str, ...] = ()


CUDA_ARCHITECTURES = {
    "sm120": CudaArchitecture(
        torch_arch="12.0a",
        min_cuda=(12, 8),
        sources=(
            "csrc/bindings.cpp",
            "csrc/cuda/sm120/nvfp4/fp4_attention.cu",
            "csrc/cuda/sm120/nvfp4/single_query_fp4_attention.cu",
            "csrc/cuda/sm120/nvfp4/thrift_attention.cu",
            "csrc/cuda/sm120/nvfp4/single_query_attention.cu",
            "csrc/cuda/sm120/nvfp4/quantization.cu",
            "csrc/cuda/sm120/mxfp4/fp4_attention.cu",
            "csrc/cuda/sm120/mxfp4/single_query_fp4_attention.cu",
            "csrc/cuda/sm120/mxfp4/thrift_attention.cu",
            "csrc/cuda/sm120/mxfp4/single_query_attention.cu",
            "csrc/cuda/sm120/mxfp4/quantization.cu",
            "csrc/cuda/sm120/shared/block_selection.cu",
        ),
        nvcc_flags=("--ptxas-options=--gpu-name=sm_120a",),
    ),
    "sm80": CudaArchitecture(
        torch_arch="8.0;8.6",
        min_cuda=(12, 8),
        sources=(
            "csrc/bindings_sm80.cpp",
            # "csrc/cuda/sm80/shared/block_selection.cu",
            "csrc/cuda/sm80/shared/mma_test.cu",
            "csrc/cuda/sm80/int8/int8_attention.cu",
            "csrc/cuda/sm80/int4/int4_attention.cu",
            "csrc/cuda/sm80/int4/quantization.cu",
        )
    )
}
DEFAULT_CUDA_ARCHITECTURE = "sm80"


def _parse_version(value: str, components: int) -> tuple[int, ...]:
    match = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", value)
    if match is None:
        return ()
    parts = [int(part) for part in match.groups(default="0")]
    return tuple(parts[:components])


def _parse_nvcc_release(value: str) -> tuple[int, int]:
    match = re.search(r"release\s+(\d+)\.(\d+)", value)
    if match is None:
        return ()
    return int(match.group(1)), int(match.group(2))


def _check_min_version(name: str, actual: tuple[int, ...], minimum: tuple[int, ...]) -> None:
    if not actual or actual < minimum:
        minimum_text = ".".join(str(part) for part in minimum)
        actual_text = ".".join(str(part) for part in actual) if actual else "unknown"
        raise RuntimeError(f"ThriftAttention requires {name}>={minimum_text}; found {actual_text}.")


def _check_build_prerequisites(architecture: CudaArchitecture) -> None:
    _check_min_version("torch", _parse_version(torch.__version__, 3), MIN_TORCH)

    torch_cuda = getattr(torch.version, "cuda", None)
    if torch_cuda is None:
        raise RuntimeError(
            "ThriftAttention requires a PyTorch CUDA wheel; "
            "the installed torch build does not report CUDA support."
        )
    torch_cuda_version = _parse_version(torch_cuda, 2)
    _check_min_version("PyTorch CUDA", torch_cuda_version, architecture.min_cuda)

    if CUDA_HOME is None:
        raise RuntimeError(
            "ThriftAttention requires a local CUDA toolkit with nvcc, "
            "but torch.utils.cpp_extension.CUDA_HOME is not set."
        )
    nvcc = Path(CUDA_HOME) / "bin" / "nvcc"
    try:
        result = subprocess.run(
            [str(nvcc), "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "ThriftAttention requires a working CUDA toolkit; "
            f"failed to run {nvcc}."
        ) from exc
    toolkit_cuda_version = _parse_nvcc_release(result.stdout)
    _check_min_version("CUDA toolkit", toolkit_cuda_version, architecture.min_cuda)
    if toolkit_cuda_version != torch_cuda_version:
        torch_cuda_text = ".".join(str(part) for part in torch_cuda_version)
        toolkit_cuda_text = ".".join(str(part) for part in toolkit_cuda_version)
        raise RuntimeError(
            "ThriftAttention builds PyTorch CUDA extensions and requires the local "
            "CUDA toolkit to match the installed PyTorch CUDA wheel. "
            f"Found nvcc CUDA {toolkit_cuda_text}, but torch was built with "
            f"CUDA {torch_cuda_text}. Set CUDA_HOME to a matching toolkit, or "
            "install a PyTorch wheel matching your local CUDA toolkit."
        )


CUDA_ARCHITECTURE = CUDA_ARCHITECTURES[DEFAULT_CUDA_ARCHITECTURE]
_check_build_prerequisites(CUDA_ARCHITECTURE)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", CUDA_ARCHITECTURE.torch_arch)


def cuda_extension(architecture: CudaArchitecture) -> CUDAExtension:
    return CUDAExtension(
        name="thriftattention._C",
        sources=list(architecture.sources),
        include_dirs=[str(ROOT / "csrc" / "include")],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": [
                "-O3",
                "-use_fast_math",
                *architecture.nvcc_flags,
            ],
        },
    )


setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[cuda_extension(CUDA_ARCHITECTURE)],
    cmdclass={"build_ext": BuildExtension},
)
