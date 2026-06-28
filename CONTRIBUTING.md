# Contributing to ThriftAttention

Thank you for your interest in contributing to Thrift Attention! This document provides guidelines and instructions for contributing to the project.

# Prerequisites

Prerequisites:

- GPU with SM120 compute capability for the full FP4/selective mixed-precision
  backend, or an SM80/SM86 GPU for the experimental dense INT8/INT4 kernels
- Python >=3.10
- CUDA toolkit >=12.8
- PyTorch >=2.8.0 built with CUDA >=12.8
- Transformers >=4.52


1. Clone the repository
```bash
git clone https://github.com/joesharratt1229/ThriftAttention.git
cd ThriftAttention
```

2. Install the package in editable mode for the target architecture
```bash
python -m pip install 'torch>=2.8.0'
THRIFTATTENTION_CUDA_ARCH=sm80 python -m pip install -e . --no-build-isolation
# Or use THRIFTATTENTION_CUDA_ARCH=sm120 on Blackwell.
```

See the installation and verification sections in `README.md` for automatic
architecture selection and architecture-specific tests.


3. Running the examples 
Check out the ReadME in `examples/` for instruction on running individual mini evaluations

4. Raising an issue. For any issues/further features please open a github issue. 
