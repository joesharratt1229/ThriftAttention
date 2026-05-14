# Video Generation Attention Stress Test

This example measures speed and agreement with an fp16/Transformers-Diffusers baseline on a fixed mini prompt set.

It is a kernel/integration stress test for long non-causal attention in Diffusers. It is not a claim that the generated videos are semantically better.

## Install

```bash
pip install -e ".[diffusers,plots]"
pip install -r examples/video_generation/requirements.txt
```

## Run

```bash
python examples/video_generation/run.py --preset quick
```

Useful overrides:

```bash
python examples/video_generation/run.py \
  --model damo-vilab/text-to-video-ms-1.7b \
  --prompts examples/video_generation/prompts_vbench_mini.jsonl \
  --methods fp16,fp4,thrift \
  --frames 8 \
  --height 256 \
  --width 256 \
  --steps 10 \
  --fraction 0.05
```

The script writes:

```text
results/video_generation/<timestamp>-video-stress/
  metrics.jsonl
  summary.md
  videos/
  environment.json
```

Reported metrics include wall-clock time, speedup versus fp16, MAE, RMSE, PSNR, and accelerated/fallback attention call counts when available.

## Presets

- `quick`: 8 frames, 256x256, few denoising steps
- `standard`: 16 frames, 256x256
- `stress`: 16 frames, 384x384

## Non-Causal Attention

This example uses `causal=False` inside the Diffusers attention processor. The core Python API has non-causal kernel dispatch, but the Diffusers integration here should be treated as experimental until it has broader model coverage.
