#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import thriftattention as ta


DEFAULT_MODEL = "damo-vilab/text-to-video-ms-1.7b"
DEFAULT_PROMPT = "A small robot painting a city skyline at sunset, cinematic, detailed"


@dataclass
class AttentionStats:
    accelerated_calls: int = 0
    fallback_calls: int = 0


class ThriftVideoAttnProcessor:
    def __init__(self, mode: str, *, fraction: float) -> None:
        if mode not in {"flash", "thrift", "fp4"}:
            raise ValueError(f"unsupported mode {mode!r}")
        self.mode = mode
        self.fraction = fraction
        self.stats = AttentionStats()

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        hidden_states, shape = _flatten_hidden_states(hidden_states)

        batch_size = hidden_states.shape[0]
        key_sequence_length = hidden_states.shape[1] if encoder_hidden_states is None else encoder_hidden_states.shape[1]
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, key_sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif getattr(attn, "norm_cross", False):
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = _heads_first(query, attn.heads)
        key = _heads_first(key, attn.heads)
        value = _heads_first(value, attn.heads)

        if self._can_use_thrift(query, key, value, attention_mask, is_cross_attention):
            if self.mode == "thrift":
                hidden_states = ta.attention(query, key, value, causal=False, fraction=self.fraction)
            else:
                hidden_states = ta.fp4_attention(query, key, value, causal=False)
            self.stats.accelerated_calls += 1
        else:
            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            self.stats.fallback_calls += 1

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, query.shape[1] * query.shape[-1])
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states = _unflatten_hidden_states(hidden_states, shape)

        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / getattr(attn, "rescale_output_factor", 1.0)
        return hidden_states

    def _can_use_thrift(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        is_cross_attention: bool,
    ) -> bool:
        if self.mode == "flash":
            return False
        if is_cross_attention or attention_mask is not None:
            return False
        if not (query.is_cuda and key.is_cuda and value.is_cuda):
            return False
        if query.dtype != torch.float16 or key.dtype != torch.float16 or value.dtype != torch.float16:
            return False
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            return False
        if query.shape[-1] not in (64, 128):
            return False
        if query.shape[2] % 64 != 0 or key.shape[2] % 64 != 0:
            return False
        return True


def _flatten_hidden_states(hidden_states: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    shape = tuple(hidden_states.shape)
    if hidden_states.ndim == 3:
        return hidden_states, shape
    if hidden_states.ndim == 4:
        batch, channels, height, width = hidden_states.shape
        return hidden_states.view(batch, channels, height * width).transpose(1, 2), shape
    if hidden_states.ndim == 5:
        batch, channels, frames, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(batch, frames * height * width, channels)
        return hidden_states, shape
    raise ValueError(f"unsupported hidden_states shape {shape}")


def _unflatten_hidden_states(hidden_states: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    if len(shape) == 3:
        return hidden_states
    if len(shape) == 4:
        batch, channels, height, width = shape
        return hidden_states.transpose(1, 2).reshape(batch, channels, height, width)
    if len(shape) == 5:
        batch, channels, frames, height, width = shape
        return hidden_states.reshape(batch, frames, height, width, channels).permute(0, 4, 1, 2, 3)
    raise ValueError(f"unsupported hidden_states shape {shape}")


def _heads_first(hidden_states: torch.Tensor, heads: int) -> torch.Tensor:
    batch, sequence_length, inner_dim = hidden_states.shape
    head_dim = inner_dim // heads
    return hidden_states.view(batch, sequence_length, heads, head_dim).transpose(1, 2).contiguous()


def _set_attention_processor(pipe: Any, processor: ThriftVideoAttnProcessor) -> None:
    patched = False
    for name in ("unet", "transformer", "transformer_3d"):
        module = getattr(pipe, name, None)
        if module is not None and hasattr(module, "set_attn_processor"):
            module.set_attn_processor(processor)
            patched = True
    if not patched:
        raise RuntimeError("pipeline has no module with set_attn_processor()")


def _flash_sdp_enabled() -> Any:
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
    except Exception:
        return contextlib.nullcontext()


def _run_once(pipe: Any, args: argparse.Namespace, mode: str) -> tuple[float, list[np.ndarray], AttentionStats]:
    processor = ThriftVideoAttnProcessor(mode, fraction=args.fraction)
    _set_attention_processor(pipe, processor)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode(), _flash_sdp_enabled():
        result = pipe(
            args.prompt,
            num_frames=args.frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            output_type="np",
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    frames = _normalise_frames(result.frames)
    return elapsed, frames, processor.stats


def _normalise_frames(frames: Any) -> list[np.ndarray]:
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().float().cpu().numpy()
    if isinstance(frames, np.ndarray):
        if frames.ndim == 5:
            frames = frames[0]
        return [_to_uint8_frame(frame) for frame in frames]
    if isinstance(frames, list) and frames and isinstance(frames[0], list):
        frames = frames[0]
    return [_to_uint8_frame(frame) for frame in frames]


def _to_uint8_frame(frame: Any) -> np.ndarray:
    if hasattr(frame, "convert"):
        frame = np.asarray(frame.convert("RGB"))
    else:
        frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0.0, 1.0)
        frame = (frame * 255.0).round().astype(np.uint8)
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.moveaxis(frame, 0, -1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return frame


def _quality(reference: list[np.ndarray], candidate: list[np.ndarray]) -> dict[str, float]:
    ref = np.stack(reference).astype(np.float32) / 255.0
    cand = np.stack(candidate).astype(np.float32) / 255.0
    diff = cand - ref
    mse = float(np.mean(diff * diff))
    rmse = math.sqrt(mse)
    psnr = float("inf") if mse == 0.0 else -10.0 * math.log10(mse)
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": rmse,
        "psnr": psnr,
    }


def _save_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    from diffusers.utils import export_to_video

    path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(path), fps=fps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end text-to-video wall-time and fp16-reference quality for ThriftAttention."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--modes", default="flash,thrift,fp4")
    parser.add_argument("--fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=9.0)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("video_generation_outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this example")

    from diffusers import TextToVideoSDPipeline

    pipe = TextToVideoSDPipeline.from_pretrained(args.model, torch_dtype=torch.float16)
    pipe = pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)

    requested_modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    unknown_modes = sorted(set(requested_modes) - {"flash", "thrift", "fp4"})
    if unknown_modes:
        raise SystemExit(f"unknown modes: {', '.join(unknown_modes)}")
    modes = ["flash"] + [mode for mode in requested_modes if mode != "flash"]

    reference_frames: list[np.ndarray] | None = None
    print("mode      wall_s   speedup  psnr_db  rmse     mae      accelerated/fallback")
    print("--------  -------  -------  -------  -------  -------  --------------------")
    for mode in modes:
        elapsed, frames, stats = _run_once(pipe, args, mode)
        _save_video(frames, args.output_dir / f"{mode}.mp4", args.fps)

        if mode == "flash":
            reference_frames = frames
            speedup = 1.0
            quality = {"psnr": float("inf"), "rmse": 0.0, "mae": 0.0}
            baseline_elapsed = elapsed
        else:
            if reference_frames is None:
                raise RuntimeError("flash reference frames were not generated")
            quality = _quality(reference_frames, frames)
            speedup = baseline_elapsed / elapsed

        psnr = "inf" if math.isinf(quality["psnr"]) else f"{quality['psnr']:.3f}"
        print(
            f"{mode:<8}  {elapsed:7.3f}  {speedup:7.3f}  {psnr:>7}  "
            f"{quality['rmse']:.5f}  {quality['mae']:.5f}  "
            f"{stats.accelerated_calls}/{stats.fallback_calls}"
        )

    print(f"\nwrote videos to {args.output_dir}")


if __name__ == "__main__":
    main()
