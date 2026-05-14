#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any


EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from common import (  # noqa: E402
    collect_environment,
    load_config,
    mae,
    make_output_dir,
    markdown_table,
    parse_str_list,
    psnr,
    rmse,
    thrift_acceleration_status,
    timed_call,
    write_json,
    write_jsonl,
)
from common.cli import pick  # noqa: E402


DEFAULT_MODEL = "damo-vilab/text-to-video-ms-1.7b"
DEFAULT_PROMPTS = Path(__file__).parent / "prompts_vbench_mini.jsonl"
VALID_METHODS = {
    "fp16": "fp16_flash",
    "flash": "fp16_flash",
    "fp16_flash": "fp16_flash",
    "fp4": "fp4",
    "thrift": "thrift",
}


@dataclass
class AttentionStats:
    accelerated_calls: int = 0
    fallback_calls: int = 0
    max_accelerated_q: int = 0
    max_accelerated_k: int = 0
    max_fallback_q: int = 0
    max_fallback_k: int = 0


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
        import torch
        import torch.nn.functional as F
        import thriftattention as ta

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
            self.stats.max_accelerated_q = max(self.stats.max_accelerated_q, query.shape[2])
            self.stats.max_accelerated_k = max(self.stats.max_accelerated_k, key.shape[2])
        else:
            hidden_states = _scaled_dot_product_attention(query, key, value, attention_mask=attention_mask)
            self.stats.fallback_calls += 1
            self.stats.max_fallback_q = max(self.stats.max_fallback_q, query.shape[2])
            self.stats.max_fallback_k = max(self.stats.max_fallback_k, key.shape[2])

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
        import torch

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Video attention stress test: speed and fp16 agreement for long non-causal Diffusers attention.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", default="quick", choices=["quick", "standard", "stress"])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompts", type=Path, default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    preset_file = {
        "quick": "quick_8frames_256.yaml",
        "standard": "standard_16frames_256.yaml",
        "stress": "stress_16frames_384.yaml",
    }[args.preset]
    config_path = args.config or Path(__file__).parent / "configs" / preset_file
    config = load_config(config_path)
    args.config_path = str(config_path)
    args.model = pick("model", args.model, config, DEFAULT_MODEL, str)
    args.prompts = Path(pick("prompts", args.prompts, config, DEFAULT_PROMPTS, Path))
    args.methods = [_normalise_method(method) for method in pick("methods", args.methods, config, ["fp16", "fp4", "thrift"], parse_str_list)]
    if "fp16_flash" not in args.methods:
        args.methods.insert(0, "fp16_flash")
    args.fraction = pick("fraction", args.fraction, config, 0.05, float)
    args.seed = pick("seed", args.seed, config, 1234, int)
    args.frames = pick("frames", args.frames, config, 8, int)
    args.height = pick("height", args.height, config, 256, int)
    args.width = pick("width", args.width, config, 256, int)
    args.steps = pick("steps", args.steps, config, 10, int)
    args.guidance_scale = pick("guidance_scale", args.guidance_scale, config, 7.5, float)
    args.fps = pick("fps", args.fps, config, 8, int)
    args.num_prompts = pick("num_prompts", args.num_prompts, config, 3, int)
    args.device = pick("device", args.device, config, "cuda", str)
    args.output = Path(pick("output", args.output, config, Path("results/video_generation"), Path))
    return args


def _normalise_method(method: str) -> str:
    key = method.strip().lower()
    if key not in VALID_METHODS:
        raise SystemExit(f"unknown method {method!r}; choose from fp16, fp4, thrift")
    return VALID_METHODS[key]


def require_video_stack() -> None:
    missing: list[str] = []
    for module in ("diffusers", "numpy"):
        try:
            __import__(module)
        except Exception:
            missing.append(module)
    if missing:
        raise SystemExit(
            "Missing optional dependencies for the video example: "
            + ", ".join(missing)
            + ". Install with `pip install -r examples/video_generation/requirements.txt` "
            "or `pip install -e '.[diffusers,plots]'`."
        )


def load_prompts(path: Path, limit: int) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = str(item.get("prompt", "")).strip()
            if not prompt:
                continue
            prompt_id = str(item.get("id", f"prompt{len(prompts) + 1:02d}"))
            prompts.append({"id": prompt_id, "prompt": prompt})
            if len(prompts) >= limit:
                break
    if not prompts:
        raise SystemExit(f"no prompts found in {path}")
    return prompts


def load_pipeline(args: argparse.Namespace) -> Any:
    import torch
    from diffusers import TextToVideoSDPipeline

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available. Use `--device cpu` for a slow fp16-only run.")
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    pipe = TextToVideoSDPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe = pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


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


def _math_sdp_enabled() -> Any:
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel([SDPBackend.MATH])
    except Exception:
        return contextlib.nullcontext()


def _scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    import torch.nn.functional as F

    effective_batches = query.shape[0] * query.shape[1]
    short_sequence = max(query.shape[2], key.shape[2]) <= 64
    context = _math_sdp_enabled() if short_sequence and effective_batches >= 4096 else contextlib.nullcontext()
    with context:
        return F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)


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


def run_once(pipe: Any, args: argparse.Namespace, prompt: str, prompt_index: int, method: str) -> tuple[float, list[Any], AttentionStats]:
    import torch

    mode = "flash" if method == "fp16_flash" else method
    processor = ThriftVideoAttnProcessor(mode, fraction=args.fraction)
    _set_attention_processor(pipe, processor)
    generator = torch.Generator(device=args.device).manual_seed(args.seed + prompt_index)

    def generate() -> Any:
        with torch.inference_mode(), _flash_sdp_enabled():
            return pipe(
                prompt,
                num_frames=args.frames,
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                output_type="np",
            )

    result, elapsed = timed_call(generate, device=args.device)
    return elapsed, normalise_frames(result.frames), processor.stats


def normalise_frames(frames: Any) -> list[Any]:
    import numpy as np
    import torch

    if isinstance(frames, torch.Tensor):
        frames = frames.detach().float().cpu().numpy()
    if isinstance(frames, np.ndarray):
        if frames.ndim == 5:
            frames = frames[0]
        return [_to_uint8_frame(frame) for frame in frames]
    if isinstance(frames, list) and frames and isinstance(frames[0], list):
        frames = frames[0]
    return [_to_uint8_frame(frame) for frame in frames]


def _to_uint8_frame(frame: Any) -> Any:
    import numpy as np

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


def frames_as_float(frames: list[Any]) -> Any:
    import numpy as np

    return np.stack(frames).astype("float32") / 255.0


def save_video(frames: list[Any], path: Path, fps: int) -> None:
    from diffusers.utils import export_to_video

    path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(path), fps=fps)


def main() -> None:
    args = resolve_args(parse_args())
    require_video_stack()
    output_dir = make_output_dir(args.output, prefix="video-stress")
    videos_dir = output_dir / "videos"
    write_json(output_dir / "environment.json", collect_environment(args))
    prompts = load_prompts(args.prompts, args.num_prompts)
    pipe = load_pipeline(args)

    thrift_ready, thrift_note = thrift_acceleration_status(args.device)
    if not thrift_ready:
        print(f"Accelerated methods will be skipped if requested: {thrift_note}")

    rows: list[dict[str, Any]] = []
    for prompt_index, item in enumerate(prompts):
        prompt_id = item["id"]
        prompt = item["prompt"]
        print(f"\nPrompt {prompt_id}: {prompt}")
        reference_frames: list[Any] | None = None
        reference_elapsed: float | None = None
        for method in args.methods:
            if method in {"fp4", "thrift"} and not thrift_ready:
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "method": method,
                        "status": "skipped",
                        "error": thrift_note,
                        "wall_s": None,
                        "speedup_vs_fp16": None,
                    }
                )
                print(f"  {method:<11} skipped: {thrift_note}")
                continue
            try:
                elapsed, frames, stats = run_once(pipe, args, prompt, prompt_index, method)
                video_path = videos_dir / f"{prompt_id}_{method}.mp4"
                save_video(frames, video_path, args.fps)
                if method == "fp16_flash":
                    reference_frames = frames
                    reference_elapsed = elapsed
                    quality = {"mae": 0.0, "rmse": 0.0, "psnr": float("inf")}
                    speedup = 1.0
                elif reference_frames is not None and reference_elapsed is not None:
                    ref = frames_as_float(reference_frames)
                    cand = frames_as_float(frames)
                    quality = {"mae": mae(ref, cand), "rmse": rmse(ref, cand), "psnr": psnr(ref, cand)}
                    speedup = reference_elapsed / elapsed
                else:
                    quality = {"mae": None, "rmse": None, "psnr": None}
                    speedup = None
                row = {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "method": method,
                    "status": "ok",
                    "wall_s": elapsed,
                    "speedup_vs_fp16": speedup,
                    "mae": quality["mae"],
                    "rmse": quality["rmse"],
                    "psnr": quality["psnr"],
                    "accelerated_calls": stats.accelerated_calls,
                    "fallback_calls": stats.fallback_calls,
                    "max_accelerated_q": stats.max_accelerated_q,
                    "max_accelerated_k": stats.max_accelerated_k,
                    "max_fallback_q": stats.max_fallback_q,
                    "max_fallback_k": stats.max_fallback_k,
                    "video": str(video_path.relative_to(output_dir)),
                    "fraction": args.fraction if method == "thrift" else None,
                }
                if method in {"fp4", "thrift"} and stats.accelerated_calls == 0:
                    row["note"] = "all attention calls used fallback"
                rows.append(row)
                psnr_text = "inf" if row["psnr"] == float("inf") else f"{row['psnr']:.3f}" if row["psnr"] is not None else "n/a"
                print(
                    f"  {method:<11} wall_s={elapsed:.3f} speedup={speedup if speedup is not None else 'n/a'} "
                    f"psnr={psnr_text} accel/fallback={stats.accelerated_calls}/{stats.fallback_calls}"
                )
            except RuntimeError as exc:
                rows.append({"prompt_id": prompt_id, "prompt": prompt, "method": method, "status": "error", "error": str(exc)})
                print(f"  {method:<11} error: {exc}")

    write_jsonl(output_dir / "metrics.jsonl", rows)
    summary_rows = build_summary_rows(rows)
    summary = "\n".join(
        [
            "# Video Attention Stress Test",
            "",
            "Speed and fp16-agreement for a small fixed prompt set. This is not a video-generation leaderboard.",
            "",
            markdown_table(
                summary_rows,
                [
                    ("method", "method"),
                    ("status", "status"),
                    ("mean_wall_s", "mean wall_s"),
                    ("mean_speedup", "mean speedup"),
                    ("mean_mae", "mean MAE"),
                    ("mean_rmse", "mean RMSE"),
                    ("mean_psnr", "mean PSNR"),
                    ("accelerated_calls", "accelerated calls"),
                    ("fallback_calls", "fallback calls"),
                ],
            ),
            "",
            f"Model: `{args.model}`",
            f"Frames/resolution: `{args.frames} @ {args.width}x{args.height}`",
            f"Videos: `videos/`",
        ]
    )
    (output_dir / "summary.md").write_text(summary + "\n", encoding="utf-8")
    print(f"\nWrote metrics.jsonl, summary.md, videos/, and environment.json under {output_dir}")


def build_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = []
    for row in rows:
        if row["method"] not in methods:
            methods.append(row["method"])
    summary: list[dict[str, Any]] = []
    for method in methods:
        items = [row for row in rows if row["method"] == method]
        ok = [row for row in items if row.get("status") == "ok"]
        if not ok:
            summary.append({"method": method, "status": items[0].get("status", "skipped") if items else "skipped"})
            continue
        summary.append(
            {
                "method": method,
                "status": "ok",
                "mean_wall_s": f"{fmean(float(row['wall_s']) for row in ok):.3f}",
                "mean_speedup": _mean_or_dash(row.get("speedup_vs_fp16") for row in ok),
                "mean_mae": _mean_or_dash(row.get("mae") for row in ok),
                "mean_rmse": _mean_or_dash(row.get("rmse") for row in ok),
                "mean_psnr": _mean_or_dash(row.get("psnr") for row in ok if not _is_inf(row.get("psnr"))),
                "accelerated_calls": sum(int(row.get("accelerated_calls", 0)) for row in ok),
                "fallback_calls": sum(int(row.get("fallback_calls", 0)) for row in ok),
            }
        )
    return summary


def _mean_or_dash(values: Any) -> str:
    clean = [float(value) for value in values if value is not None and not _is_inf(value)]
    if not clean:
        return "-"
    return f"{fmean(clean):.4g}"


def _is_inf(value: Any) -> bool:
    return isinstance(value, float) and math.isinf(value)


if __name__ == "__main__":
    main()
