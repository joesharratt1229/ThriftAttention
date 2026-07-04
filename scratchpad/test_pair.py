#!/usr/bin/env python3
"""2-CTA variant validation ladder: FORCE_PAIR=1 vs FORCE_PAIR=0 on the same
quantised inputs, shapes small -> large.  Run under `timeout`; a hang at any
shape kills the process and the last printed line names the culprit."""
import os
import sys
import math
import torch

sys.path.insert(0, "/workspace/ThriftAttention/sm100-nvfp4/sketch")
from profile_breakdown import build_variant

SHAPES = [
    # (batch, q_heads, kv_heads, q_len, kv_len)
    (1, 1, 1, 512, 512),
    (1, 1, 1, 512, 4096),
    (1, 2, 2, 512, 384),        # odd kv_iters (3)
    (1, 16, 16, 2048, 2048),    # multi-super-tile persistence
    (1, 16, 2, 2048, 1152),     # GQA 8:2 + odd iters + multi-tile
    (1, 16, 16, 8192, 8192),
]

def main() -> None:
    ext = build_variant("fp4attn_pair_ext")
    dev = torch.device("cuda")
    ok = True
    for batch, qh, kvh, q_len, kv_len in SHAPES:
        torch.manual_seed(0)
        scale = 1.0 / math.sqrt(128)
        q = torch.randn(batch, qh, q_len, 128, device=dev, dtype=torch.bfloat16)
        k = torch.randn(batch, kvh, kv_len, 128, device=dev, dtype=torch.bfloat16)
        v = torch.randn(batch, kvh, kv_len, 128, device=dev, dtype=torch.bfloat16)
        os.environ["FA4_FORCE_PAIR"] = "0"
        pre = ext.quantise_and_attention(q, k, v, scale)
        args = (pre["q_fp4"], pre["k_fp4"], pre["v_t_fp4"],
                pre["q_sf_atoms"], pre["k_sf_atoms"], pre["v_sf_atoms"])
        out0 = ext.attention_only(*args, scale)
        torch.cuda.synchronize()

        print(f"shape b{batch} qh{qh} kvh{kvh} q{q_len} kv{kv_len}: launching PAIR...",
              flush=True)
        os.environ["FA4_FORCE_PAIR"] = "1"
        out1 = ext.attention_only(*args, scale)
        torch.cuda.synchronize()
        out1b = ext.attention_only(*args, scale)
        torch.cuda.synchronize()

        exact = torch.equal(out0, out1)
        det = torch.equal(out1, out1b)
        d = (out0.float() - out1.float()).abs()
        cos = torch.nn.functional.cosine_similarity(
            out0.float().flatten(), out1.float().flatten(), dim=0).item()
        fin = torch.isfinite(out1.float()).all().item()
        print(f"  pair-vs-1cta bitexact={exact} deterministic={det} finite={fin} "
              f"cos={cos:.6f} maxdiff={d.max().item():.4f}", flush=True)
        ok &= det and fin and (exact or cos > 0.9999)
    print("PAIR LADDER:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
