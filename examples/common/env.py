from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any


def collect_environment(args: Namespace | dict[str, Any] | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "command": sys.argv,
        "packages": {},
    }
    for package in (
        "torch",
        "transformers",
        "datasets",
        "diffusers",
        "accelerate",
        "matplotlib",
        "thriftattention",
    ):
        env["packages"][package] = _version(package)

    try:
        import torch
    except Exception as exc:
        env["torch_error"] = repr(exc)
    else:
        env["torch"] = {
            "version": torch.__version__,
            "cuda_version": getattr(torch.version, "cuda", None),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            env["gpus"] = []
            for index in range(torch.cuda.device_count()):
                env["gpus"].append(
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "capability": list(torch.cuda.get_device_capability(index)),
                        "total_memory_gb": round(torch.cuda.get_device_properties(index).total_memory / 1024**3, 3),
                    }
                )

    if args is not None:
        env["args"] = _json_safe(vars(args) if isinstance(args, Namespace) else args)
    if extra:
        env.update(_json_safe(extra))
    return env


def thrift_acceleration_status(device: str = "cuda") -> tuple[bool, str]:
    try:
        import torch
    except Exception as exc:
        return False, f"PyTorch is not importable: {exc}"

    if not device.startswith("cuda"):
        return False, "ThriftAttention FP4 kernels require CUDA tensors."
    if not torch.cuda.is_available():
        return False, "CUDA is not available; fp16/standard attention can still run on CPU."

    cuda_index = _cuda_index(device)
    major, minor = torch.cuda.get_device_capability(cuda_index)
    if (major, minor) != (12, 0):
        name = torch.cuda.get_device_name(cuda_index)
        return (
            False,
            f"Current public ThriftAttention kernels target SM120 consumer Blackwell; "
            f"device {cuda_index} is {name} with capability sm_{major}{minor}.",
        )

    try:
        from thriftattention._extension import get_extension

        get_extension()
    except Exception as exc:
        return False, f"ThriftAttention CUDA extension is unavailable or failed to load: {exc}"
    return True, "ThriftAttention SM120 CUDA extension is available."


def _cuda_index(device: str) -> int:
    if ":" not in device:
        return 0
    try:
        return int(device.split(":", 1)[1])
    except ValueError:
        return 0


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        return str(value)
