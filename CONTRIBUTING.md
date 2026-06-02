# Contributing to ThriftAttention

Thank you for your interest in contributing to Thrift Attention! This document provides guidelines and instructions for contributing to the project.

# Prerequisites

Prerequisites:

- GPU with SM_120 compute capability e.g. RTX 5090, RTX 6000, RTX 4500. Ampere
  SM_80 is registered as a backend target, but its kernels are not implemented yet.
- Python >=3.10
- CUDA toolkit >=12.8
- PyTorch >=2.8.0 built with CUDA >=12.8
- Transformers >=4.52


1. Clone the repository
```bash
git clone https://github.com/joesharratt1229/ThriftAttention.git
cd ThriftAttention
```

2. Install the package in editable mode
```bash
python -m pip install 'torch>=2.8.0'
python -m pip install -e . --no-build-isolation
```


3. Running the examples 
Check out the ReadME in `examples/` for instruction on running individual mini evaluations

4. Raising an issue. For any issues/further features please open a github issue. 
