from pathlib import Path
import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).parent
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")


def cuda_extension() -> CUDAExtension:
    sources = [
        "csrc/bindings.cpp",
        "csrc/cuda/sm120/nvfp4/fp4_attention.cu",
        "csrc/cuda/sm120/nvfp4/thrift_attention.cu",
        "csrc/cuda/sm120/nvfp4/quantization.cu",
        "csrc/cuda/sm120/nvfp4/block_selection.cu",
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
                "--ptxas-options=--gpu-name=sm_120a",
            ],
        },
    )


setup(
    name="thriftattention",
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[cuda_extension()],
    cmdclass={"build_ext": BuildExtension},
)
