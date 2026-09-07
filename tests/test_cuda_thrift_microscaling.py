"""Regression checks for Thrift's P microscaling and mixed-precision handoff.

Run directly with: python tests/test_cuda_thrift_microscaling.py
Also collected by pytest when available.
"""
from __future__ import annotations

import itertools
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from thriftattention._extension import get_extension
from thriftattention.quant.formats import get_quant_format


def dequantize(data: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=data.device)
    codes = torch.stack((data & 15, data >> 4), dim=-1).flatten(-2).long()
    return levels[codes & 7] * torch.where(codes < 8, 1, -1) * scales.float().repeat_interleave(16, -1)


def fp4_round(x: torch.Tensor) -> torch.Tensor:
    # At a midpoint, choose the even E2M1 code, as CUDA cvt.rn does.
    levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=x.device)
    priority = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=x.device)
    candidates = levels[priority]
    return candidates[(x.unsqueeze(-1) - candidates).abs().argmin(-1)]


def mixed_reference(q, k, v, packed, selected, causal, approx=False):
    """Logical-order tensor reference, independent of MMA fragment layout."""
    q4, k4, vt4 = (dequantize(packed[i], packed[i + 3]) for i in range(3))
    n = k.shape[2]
    physical = torch.arange(n, device=q.device)
    x = physical % 32
    logical = physical // 32 * 32 + x // 8 * 2 + (x % 8) // 2 * 8 + x % 2
    k4 = k4[:, :, logical.argsort()]
    v4 = vt4.transpose(-2, -1)[:, :, :n]
    groups = q.shape[1] // k.shape[1]
    k4, v4, k, v = (x.repeat_interleave(groups, 1).float() for x in (k4, v4, k, v))
    result = torch.empty_like(q)
    for qi in range(q.shape[2] // 64):
        query = slice(qi * 64, (qi + 1) * 64)
        shape = (*q.shape[:2], 64, 1)
        m = torch.full(shape, -126.0 if approx else torch.finfo(torch.float32).min, device=q.device)
        ell = torch.zeros_like(m)
        out = torch.zeros((*shape[:-1], q.shape[-1]), device=q.device)
        valid_blocks = list(range(qi + 1 if causal else n // 64))
        low_blocks = [j for j in valid_blocks if j not in selected]
        high_blocks = [j for j in selected if j in valid_blocks]
        for high, blocks in ((False, low_blocks), (True, high_blocks)):
            for j in blocks:
                keys = slice(j * 64, (j + 1) * 64)
                scores = ((q[:, :, query].float() if high else q4[:, :, query])
                          @ (k[:, :, keys] if high else k4[:, :, keys]).transpose(-2, -1))
                scores *= q.shape[-1] ** -.5 * (1.4426950408889634 if approx and not high else 1.0)
                if causal:
                    qpos = torch.arange(qi * 64, (qi + 1) * 64, device=q.device)
                    kpos = torch.arange(j * 64, (j + 1) * 64, device=q.device)
                    scores.masked_fill_(kpos[None, :] > qpos[:, None], -torch.inf)
                tile_max = scores.amax(-1, keepdim=True)
                if approx and not high:
                    tile_max = (tile_max + .5).round()
                new_m = torch.maximum(m, tile_max)
                rescale = (m - new_m).exp2() if approx and not high else (m - new_m).exp()
                out *= rescale
                ell *= rescale
                m = new_m
                p = (scores - m).exp()
                if not approx or high:
                    ell += p.sum(-1, keepdim=True)
                if high:
                    weights = (2688 * p).to(q.dtype).float()
                elif approx:
                    blocks_s = scores.unflatten(-1, (4, 16))
                    b = (blocks_s.amax(-1, keepdim=True) + .5).round().clamp_min(-126)
                    sf = (448 * (b - m.unsqueeze(-1)).exp2()).to(torch.float8_e4m3fn).float()
                    weights = (fp4_round(4 * (blocks_s - b).round().exp2()) * sf).flatten(-2)
                    ell += weights.sum(-1, keepdim=True) / 1792
                else:
                    blocks_s = scores.unflatten(-1, (4, 16))
                    b = blocks_s.amax(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).min)
                    sf = (448 * (b - m.unsqueeze(-1)).exp()).to(torch.float8_e4m3fn).float()
                    weights = (fp4_round(6 * (blocks_s - b).exp()) * sf).flatten(-2)
                out += weights @ (v[:, :, keys] if high else v4[:, :, keys])
            if not high:
                # The two kernels exchange a normalized FP16/BF16 partial.
                partial = (out / ((1792 if approx else 2688) * ell.clamp_min(1e-30))).to(q.dtype).float()
                if approx:
                    m = torch.where(ell > 0, m * 0.6931471805599453, torch.finfo(torch.float32).min)
                out = partial * (2688 * ell)
        result[:, :, query] = out / (2688 * ell)
    return result


class ThriftMicroscalingTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    @torch.no_grad()
    def test_fp4_and_mixed_selections(self):
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("SM120 GPU required")
        ext = get_extension()
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", old_tf32)
        for dtype, dim, causal, approx in itertools.product(
            (torch.float16, torch.bfloat16), (64, 128), (False, True), (False, True)
        ):
            torch.manual_seed(41)
            q = torch.zeros((2, 2, 192, dim), dtype=dtype, device="cuda")
            q_values = torch.tensor([0, .5, 1, 2, 4, 6, -2], device="cuda", dtype=dtype)
            q[..., :16] = q_values[torch.arange(192, device="cuda") % 7][None, None, :, None]
            k = torch.zeros((2, 1, 192, dim), dtype=dtype, device="cuda")
            # Independent microblock maxima, weak tails, and a rising maximum
            # across the three KV tiles. Q's extra channels are zero.
            k_values = torch.tensor([1.5, .5, 1, 2, .5, 3, 1, 4, 1, 6, 2, .5], device="cuda", dtype=dtype)
            k[..., :16] = (k_values.repeat_interleave(16) * .125)[None, None, :, None]
            k[..., 16:32] = .75
            v = torch.randn_like(k)
            packed = get_quant_format("nvfp4").quantize_qkv(q, k, v, is_bf16=dtype == torch.bfloat16)
            fn = ext.thrift_attention_causal_nvfp4_packed if causal else ext.thrift_attention_noncausal_nvfp4_packed
            for chosen in ([], [0], [1], [2], [0, 1, 2]):
                with self.subTest(dtype=dtype, dim=dim, causal=causal, selected=chosen, approx=approx):
                    # A -1 sentinel exercises Thrift's FP4 pass directly;
                    # top_k=0 would dispatch the standalone FP4 kernel.
                    selected = torch.tensor(chosen or [-1], device="cuda", dtype=torch.int32)
                    selected = selected.view(1, 1, -1).expand(4, 3, -1).contiguous()
                    actual = fn(q, k, v, selected, *packed, dtype == torch.bfloat16, approx)
                    if approx and chosen == [0, 1, 2]:
                        baseline = fn(q, k, v, selected, *packed, dtype == torch.bfloat16, False)
                        torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
                    if not chosen:
                        empty = selected[:, :, :0].contiguous()
                        fallback = fn(q, k, v, empty, *packed, dtype == torch.bfloat16, approx)
                        pure_fn = ext.fp4_attention_causal_nvfp4_packed if causal else ext.fp4_attention_noncausal_nvfp4_packed
                        pure = pure_fn(*packed, dtype == torch.bfloat16, approx)
                        torch.testing.assert_close(fallback, pure, rtol=0, atol=0)
                    expected = mixed_reference(q, k, v, packed, chosen, causal, approx)
                    torch.cuda.synchronize()
                    self.assertTrue(torch.isfinite(actual).all().item())
                    torch.testing.assert_close(actual.float(), expected.float(),
                                               atol=.002 if dtype == torch.bfloat16 else .0003,
                                               rtol=.01 if dtype == torch.bfloat16 else .002)


if __name__ == "__main__":
    unittest.main()
