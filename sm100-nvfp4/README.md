# SM100 NVFP4 quantise smoke test

Run the local quantisation kernel on a Modal B200:

```bash
modal run modal_app.py
```

Useful knobs:

```bash
modal run modal_app.py --bs 1 --seq-len 128 --head-dim 64 --arch sm_100a
```

The Modal function uses `nvidia/cuda:12.8.1-devel-ubuntu24.04`, requests
`gpu="B200"`, compiles `quantise_nvfp4.cu` with `nvcc -arch=sm_100a`, and runs
`quantise_smoke.cu` against all four launcher wrappers in this directory.
