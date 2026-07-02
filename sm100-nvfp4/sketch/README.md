# SM100 NVFP4 attention smoke test

Run quantisation first, then compile and run the SM100 NVFP4 attention smoke
test on a Modal B200:

```bash
modal run sketch/modal_app.py
```

Useful knobs:

```bash
modal run sketch/modal_app.py --bs 1 --q-len 256 --kv-len 128 --num-q-heads 1 --num-kv-heads 1 --arch sm_100a
```

The Modal function uses `nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04`, requests
`gpu="B200"`, compiles `quantise_nvfp4.cu` with
`-gencode=arch=compute_100a,code=sm_100a`, runs `quantise_smoke.cu` against the
quantisation launchers, then runs
`fp4_attention_sm100.py`. That script compiles `fp4_attention_sm100_smoke.cu`
with `quantise_nvfp4.cu` and `fp4_attention_sm100.cu`, packs Q/K/V scale-factor
atoms, launches `nvfp4_sm100_attention_launch`, and prints a bf16 output sample.

Current smoke-test constraint: `q_len` must be divisible by 256 and `kv_len`
must be divisible by 128. The kernel only requires `kv_len % 64 == 0`, but the
existing V transpose quantizer pads to 128-row sequence blocks.
