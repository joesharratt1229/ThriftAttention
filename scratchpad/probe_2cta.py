"""Probe: how does a cta_group::2 tiled MMA partition A/B/C across the peer CTAs?

Constructs the same tiled MMAs FA4 uses (QK: K-major x K-major; PV: tmem A,
MN-major B) with CtaGroup.TWO, M=256, and prints static partition shapes and
per-peer coordinate slices at trace time.
"""
import os
os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, BFloat16

M_BLOCK = 128
N_BLOCK = 128
HEAD_DIM = 128
CTA_GROUP_SIZE = 2

mma_tiler_qk = (CTA_GROUP_SIZE * M_BLOCK, N_BLOCK, HEAD_DIM)
mma_tiler_pv = (CTA_GROUP_SIZE * M_BLOCK, HEAD_DIM, N_BLOCK)


@cute.jit
def probe():
    cta_group = tcgen05.CtaGroup.TWO
    tiled_mma_qk = sm100_utils.make_trivial_tiled_mma(
        BFloat16,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
        Float32,
        cta_group,
        mma_tiler_qk[:2],
    )
    tiled_mma_pv = sm100_utils.make_trivial_tiled_mma(
        BFloat16,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.MN,
        Float32,
        cta_group,
        mma_tiler_pv[:2],
        tcgen05.OperandSource.TMEM,
    )
    print("thr_id qk:", tiled_mma_qk.thr_id)
    print("QK partition_shape_A (M,K):",
          tiled_mma_qk.partition_shape_A((mma_tiler_qk[0], mma_tiler_qk[2])))
    print("QK partition_shape_B (N,K):",
          tiled_mma_qk.partition_shape_B((mma_tiler_qk[1], mma_tiler_qk[2])))
    print("QK partition_shape_C (M,N):",
          tiled_mma_qk.partition_shape_C(mma_tiler_qk[:2]))
    print("PV partition_shape_A (M,K):",
          tiled_mma_pv.partition_shape_A((mma_tiler_pv[0], mma_tiler_pv[2])))
    print("PV partition_shape_B (N,K):",
          tiled_mma_pv.partition_shape_B((mma_tiler_pv[1], mma_tiler_pv[2])))
    print("PV partition_shape_C (M,N):",
          tiled_mma_pv.partition_shape_C(mma_tiler_pv[:2]))

    # Identity layouts so partition coordinates are readable.
    identA = cute.make_identity_layout((mma_tiler_qk[0], mma_tiler_qk[2]))
    identB = cute.make_identity_layout((mma_tiler_qk[1], mma_tiler_qk[2]))
    identC = cute.make_identity_layout(mma_tiler_qk[:2])
    tA = tiled_mma_qk._thrfrg_A(identA)
    tB = tiled_mma_qk._thrfrg_B(identB)
    tC = tiled_mma_qk._thrfrg_C(identC)
    print("QK _thrfrg_A:", tA)
    print("QK _thrfrg_B:", tB)
    print("QK _thrfrg_C:", tC)

    identB_pv = cute.make_identity_layout((mma_tiler_pv[1], mma_tiler_pv[2]))
    print("PV _thrfrg_B:", tiled_mma_pv._thrfrg_B(identB_pv))


probe()
print("OK")
