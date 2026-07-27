from pathlib import Path
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
MIN_CUDA = (12, 8)


def parse_version(value: str, components: int) -> tuple[int, ...]:
    match = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", value)
    if match is None:
        return ()
    parts = [int(part) for part in match.groups(default="0")]
    return tuple(parts[:components])


def parse_nvcc_release(value: str) -> tuple[int, int]:
    match = re.search(r"release\s+(\d+)\.(\d+)", value)
    if match is None:
        return ()
    return int(match.group(1)), int(match.group(2))


def check_min_version(name: str, actual: tuple[int, ...], minimum: tuple[int, ...]) -> None:
    if not actual or actual < minimum:
        minimum_text = ".".join(str(part) for part in minimum)
        actual_text = ".".join(str(part) for part in actual) if actual else "unknown"
        raise RuntimeError(f"ThriftAttention requires {name}>={minimum_text}; found {actual_text}.")


def check_build_prerequisites() -> None:
    check_min_version("torch", parse_version(torch.__version__, 3), MIN_TORCH)

    torch_cuda = getattr(torch.version, "cuda", None)
    if torch_cuda is None:
        raise RuntimeError(
            "ThriftAttention requires a PyTorch CUDA wheel built with CUDA>=12.8; "
            "the installed torch build does not report CUDA support."
        )
    torch_cuda_version = parse_version(torch_cuda, 2)
    check_min_version("PyTorch CUDA", torch_cuda_version, MIN_CUDA)

    if CUDA_HOME is None:
        raise RuntimeError(
            "ThriftAttention requires a local CUDA toolkit with nvcc>=12.8, "
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
            f"ThriftAttention requires a working CUDA toolkit with nvcc>=12.8; "
            f"failed to run {nvcc}."
        ) from exc
    toolkit_cuda_version = parse_nvcc_release(result.stdout)
    check_min_version("CUDA toolkit", toolkit_cuda_version, MIN_CUDA)
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


def default_cuda_arch_list() -> str:
    try:
        if torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0):
            return "10.0a"
    except Exception:
        pass
    return "12.0a"


os.environ.setdefault("TORCH_CUDA_ARCH_LIST", default_cuda_arch_list())
ARCH_LIST = os.environ["TORCH_CUDA_ARCH_LIST"]
ARCH_TOKENS = tuple(part.strip().lower() for part in re.split(r"[;,\s]+", ARCH_LIST) if part.strip())
BUILDS_SM100 = any(
    token.startswith("10.0a") or token.startswith("sm_100a") or token.startswith("compute_100a")
    for token in ARCH_TOKENS
)
BUILDS_SM120 = any(
    token.startswith("12.0a") or token.startswith("sm_120a") or token.startswith("compute_120a")
    for token in ARCH_TOKENS
)

if any(token in {"10.0", "sm_100", "compute_100"} for token in ARCH_TOKENS):
    raise RuntimeError(
        "SM100 FP4 kernels require architecture-specific codegen. "
        "Use TORCH_CUDA_ARCH_LIST=10.0a so nvcc emits compute_100a -> sm_100a."
    )
if BUILDS_SM100 and BUILDS_SM120:
    raise RuntimeError(
        "ThriftAttention does not yet build SM100 and SM120 kernels into one extension. "
        "Use TORCH_CUDA_ARCH_LIST=10.0a for SM100 or 12.0a for SM120."
    )
if BUILDS_SM100 or BUILDS_SM120:
    check_build_prerequisites()


PTXAS_GPU_NAME = os.environ.get(
    "THRIFTATTENTION_PTXAS_GPU_NAME",
    "sm_100a" if BUILDS_SM100 else "sm_120a",
)


def cuda_extension() -> CUDAExtension:
    sources = [
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
    ]
    if BUILDS_SM100:
        sources = [
            "csrc/bindings_sm100.cpp",
            "csrc/cuda/sm120/nvfp4/quantization.cu",
            "csrc/cuda/sm100/nvfp4/fp4_attention.cu",
        ]
    return CUDAExtension(
        name="thriftattention._C",
        sources=sources,
        include_dirs=[str(ROOT / "csrc" / "include")],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": [
                "-O3",
                "-use_fast_math",
                f"--ptxas-options=--gpu-name={PTXAS_GPU_NAME}",
            ],
        },
    )


setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[cuda_extension()],
    cmdclass={"build_ext": BuildExtension},
)
