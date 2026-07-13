#!/usr/bin/env python3
"""Where does nvfp4_sm100_attention_kernel spend its time?

Two independent methods (hardware counters are blocked in this container):

1. Default: globaltimer instrumentation.  The kernel is rebuilt with
   -DFA4_PROF=1; every specialised warp accumulates per-segment nanoseconds
   (each wait vs each piece of work) and dumps them per CTA at kernel end.
   The report shows, per warp role, the fraction of that warp's lifetime in
   each segment.  Instrumentation overhead is measured against the
   uninstrumented build and reported.

2. --ablate: counterfactual ablations.  The kernel is rebuilt with one piece
   of work compiled out (S loads / max reduce / convert / rescale / KV TMA
   data / ALL softmax work) while the full mbarrier protocol is preserved,
   then interleave-timed against the production build.  "shell" = the time
   left when all softmax work is removed = MMA + TMA + mbarrier chain +
   handshake latencies (the RESTRUCTURE_PLAN 53% number).

Reading the mma-warp view:
  waiting KVFull       -> TMA/DRAM bound
  waiting SEmpty/P0/P1 -> softmax bound
  issue+commit         -> MMA issue bound
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent
HEAD_DIM = 128

BASE_CUDA_FLAGS = [
    "-O3",
    "-std=c++17",
    "-gencode=arch=compute_100a,code=sm_100a",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
    "--relocatable-device-code=false",
    "-lineinfo",
]

# Slot layout (must mirror fp4_attention_sm100.cu): warp_id 0-3 = softmax
# owners, 4-7 = partners, 8 = mma, 9 = load, 10 = store; cat 11 = warp span.
# Values are SM cycle counts; only fractions of the span are meaningful.
# The default (coarse, FA4_PROF_SM_LEVEL=1) build records only wait-vs-busy
# for the softmax warps; cats 1-2 stay zero and cat 3 covers all busy work.
SPAN = 11
ROLES = [
    ("softmax owner (w0-3)", slice(0, 4), [
        (0, "wait SFull (S ready: QK mma+chain)"),
        (1, "S tmem load (tcgen05.ld x64)"),
        (2, "stats (reduce+pair bars+rescale)"),
        (3, "busy (load+stats+convert+store P)"),
        (4, "epilogue (OFull/OStore waits+norm)"),
    ]),
    ("softmax partner (w4-7)", slice(4, 8), [
        (0, "wait SFull (S ready: QK mma+chain)"),
        (1, "S tmem load (tcgen05.ld x64)"),
        (2, "stats (reduce+pair bars)"),
        (3, "busy (load+stats+convert+store P)"),
    ]),
    ("mma (w8)", slice(8, 9), [
        (0, "wait QFull (Q TMA, per tile)"),
        (1, "wait KVFull K (K TMA)"),
        (2, "wait SEmpty (S buf recycled)"),
        (3, "QK issue (sf cp+mma+commits)"),
        (4, "wait KVFull V (V TMA)"),
        (5, "wait P0Ready (owner convert)"),
        (6, "wait P1Ready (partner convert)"),
        (7, "wait OEmpty (epilogue frees O)"),
        (8, "PV issue (sf cp+mma+commits)"),
    ]),
    ("load (w10)", slice(9, 10), [
        (0, "wait QEmpty"),
        (1, "wait KVEmpty (K slot)"),
        (2, "issue K TMA"),
        (3, "wait KVEmpty (V slot)"),
        (4, "issue V TMA"),
        (5, "issue Q TMA"),
    ]),
    ("store (w9)", slice(10, 11), [
        (0, "wait OEpi (owners staged O)"),
        (1, "O TMA store + drain + arrive"),
    ]),
]

ABLATIONS = [
    ("sload",   ["FA4_ABLATE_SLOAD=1"],   "S tmem score loads"),
    ("redmax",  ["FA4_ABLATE_REDMAX=1"],  "block max reduce"),
    ("convert", ["FA4_ABLATE_CONVERT=1"], "score->P convert + P/sf stores"),
    ("rescale", ["FA4_ABLATE_RESCALE=1"], "O rescale (correction)"),
    ("kvtma",   ["FA4_ABLATE_KV_TMA=1"],  "K/V TMA data fetch (DRAM traffic)"),
    ("shell",   ["FA4_ABLATE_SLOAD=1", "FA4_ABLATE_REDMAX=1",
                 "FA4_ABLATE_CONVERT=1", "FA4_ABLATE_RESCALE=1"],
     "ALL softmax work (remainder = shell)"),
]


def build_variant(name: str, defines: list[str] | tuple = (), verbose: bool = False):
    build_dir = ROOT / ".torch_extensions" / name
    build_dir.mkdir(parents=True, exist_ok=True)
    # Killed builds leave a stale file baton that blocks all later builds
    # forever (see RESTRUCTURE_PLAN.txt ops note) -- always clear it.
    (build_dir / "lock").unlink(missing_ok=True)
    return load(
        name=name,
        sources=[
            str(ROOT / "fp4_attention_extension.cpp"),
            str(ROOT / "fp4_attention_sm100.cu"),
            str(ROOT / "quantise_nvfp4.cu"),
        ],
        extra_cuda_cflags=BASE_CUDA_FLAGS + [f"-D{d}" for d in defines],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=["-lcuda"],
        build_directory=str(build_dir),
        verbose=verbose,
    )


def make_inputs(ext, batch: int, heads: int, q_len: int, kv_len: int):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    q = torch.randn(batch, heads, q_len, HEAD_DIM, device="cuda", dtype=dtype)
    k = torch.randn(batch, heads, kv_len, HEAD_DIM, device="cuda", dtype=dtype)
    v = torch.randn(batch, heads, kv_len, HEAD_DIM, device="cuda", dtype=dtype)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    pre = ext.quantise_and_attention(q, k, v, scale)
    args = (pre["q_fp4"], pre["k_fp4"], pre["v_t_fp4"],
            pre["q_sf_atoms"], pre["k_sf_atoms"], pre["v_sf_atoms"])
    return args, scale


def time_fn(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def time_interleaved(fn_a, fn_b, warmup: int, iters: int, chunks: int = 10):
    """A/B in alternating chunks so both see the same thermal state."""
    for _ in range(warmup):
        fn_a()
        fn_b()
    chunk = max(1, iters // chunks)
    total_a = total_b = 0.0
    n = 0
    for _ in range(chunks):
        for which, fn in (("a", fn_a), ("b", fn_b)):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            torch.cuda.synchronize()
            s.record()
            for _ in range(chunk):
                fn()
            e.record()
            torch.cuda.synchronize()
            if which == "a":
                total_a += s.elapsed_time(e)
            else:
                total_b += s.elapsed_time(e)
        n += chunk
    return total_a / n, total_b / n


def profiled_breakdown(prof_ext, args, scale, warmup: int, reps: int):
    for _ in range(warmup):
        prof_ext.attention_only(*args, scale)
    torch.cuda.synchronize()
    _, ctas, slots, cats = prof_ext.prof_info()
    acc = torch.zeros(ctas, slots, cats, dtype=torch.float64)
    wall = 0.0
    for _ in range(reps):
        prof_ext.prof_reset()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        prof_ext.attention_only(*args, scale)
        e.record()
        torch.cuda.synchronize()
        wall += s.elapsed_time(e)
        acc += prof_ext.prof_fetch().cpu().double()
    return acc / reps, wall / reps


def print_breakdown(acc: torch.Tensor, wall_ms: float, flops: float):
    print(f"  kernel wall: {wall_ms:.3f} ms  ({flops / (wall_ms * 1e-3) / 1e12:.0f} TF/s)")
    summary = {}
    for role_name, sl, cat_list in ROLES:
        role = acc[:, sl, :]                     # [ctas, warps, cats]
        span = role[:, :, SPAN]
        active = span > 0
        n_active = int(active.sum().item())
        if n_active == 0:
            print(f"\n  -- {role_name}: no samples --")
            continue
        span_total = float(span.sum().item())
        # Spans are SM cycles; implied clock is a sanity check that the
        # warp lives for ~the whole kernel.
        ghz = span_total / n_active / (wall_ms * 1e6)
        print(f"\n  -- {role_name}: span/wall implies {ghz:.2f} GHz SM clock --")
        cat_sum = 0.0
        fracs = {}
        for ci, label in cat_list:
            cyc = float(role[:, :, ci].sum().item())
            cat_sum += cyc
            frac = cyc / span_total
            fracs[ci] = frac
            print(f"    {label:<38} {frac * 100:6.1f}%   {frac * wall_ms * 1e3:9.1f} us")
        gap = 1.0 - cat_sum / span_total
        print(f"    {'(untracked: loop overhead + tail)':<38} {gap * 100:6.1f}%")
        summary[role_name] = fracs
    return summary


def print_derived(summary: dict):
    own = summary.get("softmax owner (w0-3)")
    par = summary.get("softmax partner (w4-7)")
    mma = summary.get("mma (w8)")
    if not (own and par and mma):
        return
    print("\n  == derived (softmax view: owner+partner averaged) ==")
    sm = {k: (own.get(k, 0.0) + par.get(k, 0.0)) / 2 for k in set(own) | set(par)}
    print(f"    wait for S (QK mma+TMA+barrier shell):    {sm.get(0, 0) * 100:5.1f}%")
    print(f"    softmax busy (S-load+stats+convert):      {(sm.get(1, 0) + sm.get(2, 0) + sm.get(3, 0)) * 100:5.1f}%")
    if sm.get(1, 0) + sm.get(2, 0) > 0:
        print(f"      of which  S tmem load:                  {sm.get(1, 0) * 100:5.1f}%")
        print(f"      of which  stats/pair-sync/rescale:      {sm.get(2, 0) * 100:5.1f}%")
        print(f"      of which  convert+store P:              {sm.get(3, 0) * 100:5.1f}%")
    print(f"    epilogue (waits+norm, owners only):       {own.get(4, 0) * 100:5.1f}%")
    print("\n  == derived (mma-warp view: what gates the next MMA) ==")
    tma = mma.get(1, 0) + mma.get(4, 0)
    smx = mma.get(2, 0) + mma.get(5, 0) + mma.get(6, 0)
    issue = mma.get(3, 0) + mma.get(8, 0)
    tile = mma.get(0, 0) + mma.get(7, 0)
    print(f"    waiting on TMA (KVFull K+V):        {tma * 100:5.1f}%")
    print(f"    waiting on softmax (SEmpty+P0+P1):  {smx * 100:5.1f}%")
    print(f"    issue+commit work (QK+PV):          {issue * 100:5.1f}%")
    print(f"    tile chain (QFull+OEmpty):          {tile * 100:5.1f}%")


def run_instrumented(shapes, batch, heads, warmup, reps, iters, fine=False):
    print("building production extension (baseline)...")
    prod = build_variant("fp4_attention_sm100_ext")
    if fine:
        print("building FA4_PROF=1 (fine softmax ticks) extension...")
        prof = build_variant("fp4attn_prof_ext", ["FA4_PROF=1", "FA4_PROF_SM_LEVEL=2"])
    else:
        print("building FA4_PROF=1 (coarse softmax ticks) extension...")
        prof = build_variant("fp4attn_prof1_ext", ["FA4_PROF=1", "FA4_PROF_SM_LEVEL=1"])
    enabled = prof.prof_info()[0]
    assert enabled == 1, "prof build does not report FA4_PROF=1"

    for s in shapes:
        q_len = kv_len = s
        print(f"\n=== shape: batch={batch} heads={heads} q=kv={s} ===")
        args, scale = make_inputs(prod, batch, heads, q_len, kv_len)

        # Sanity: instrumentation must not change the math.
        out_prod = prod.attention_only(*args, scale)
        out_prof = prof.attention_only(*args, scale)
        bitexact = torch.equal(out_prod, out_prof)
        print(f"  prof output bit-exact vs production: {bitexact}")

        # Perturbation: how much does the instrumentation slow the kernel?
        prod_ms, prof_ms = time_interleaved(
            lambda: prod.attention_only(*args, scale),
            lambda: prof.attention_only(*args, scale),
            warmup, iters)
        print(f"  production {prod_ms:.3f} ms | instrumented {prof_ms:.3f} ms "
              f"(overhead {100 * (prof_ms / prod_ms - 1):+.1f}%)")

        acc, wall_ms = profiled_breakdown(prof, args, scale, warmup, reps)
        flops = 4.0 * batch * heads * q_len * kv_len * HEAD_DIM
        summary = print_breakdown(acc, wall_ms, flops)
        print_derived(summary)
        del args
        torch.cuda.empty_cache()


def run_ablations(shapes, batch, heads, warmup, iters):
    print("building production extension (baseline)...")
    prod = build_variant("fp4_attention_sm100_ext")
    variants = []
    for key, defines, label in ABLATIONS:
        print(f"building ablation '{key}' ({label})...")
        variants.append((key, label, build_variant(f"fp4attn_ab_{key}_ext", defines)))

    for s in shapes:
        q_len = kv_len = s
        print(f"\n=== ablations: batch={batch} heads={heads} q=kv={s} ===")
        args, scale = make_inputs(prod, batch, heads, q_len, kv_len)
        print(f"  {'ablated work':<38} {'full ms':>9} {'ablated ms':>10} "
              f"{'delta':>7} {'share':>6}")
        for key, label, ext in variants:
            full_ms, abl_ms = time_interleaved(
                lambda: prod.attention_only(*args, scale),
                lambda: ext.attention_only(*args, scale),
                warmup, iters)
            delta = full_ms - abl_ms
            print(f"  {label:<38} {full_ms:>9.3f} {abl_ms:>10.3f} "
                  f"{delta / full_ms * 100:>6.1f}% {abl_ms / full_ms * 100:>5.1f}%")
        print("  ('share' of the 'shell' row = the RESTRUCTURE_PLAN 53% number;")
        print("   deltas overlap, so they need not sum to 100%.)")
        del args
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seqlens", type=int, nargs="+", default=[8192])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=8,
                    help="instrumented launches averaged for the breakdown")
    ap.add_argument("--iters", type=int, default=50,
                    help="timed iterations for wall-clock comparisons")
    ap.add_argument("--ablate", action="store_true",
                    help="also run counterfactual ablation builds")
    ap.add_argument("--fine", action="store_true",
                    help="fine-grained softmax ticks (higher perturbation)")
    args = ap.parse_args()

    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name()}")
    run_instrumented(args.seqlens, args.batch, args.heads,
                     args.warmup, args.reps, args.iters, fine=args.fine)
    if args.ablate:
        run_ablations(args.seqlens, args.batch, args.heads,
                      args.warmup, args.iters)


if __name__ == "__main__":
    main()
