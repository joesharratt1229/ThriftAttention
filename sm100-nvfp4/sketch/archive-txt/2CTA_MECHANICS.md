# 2-CTA cluster MMA mechanics — extraction report (2026-07-04)

Evidence sources (all on this box):
- `FA4SRC` = /workspace/ThriftAttention/sm100-nvfp4/sketch/helpers/flash_fwd_sm100.py (same as installed flash_attn/cute/flash_fwd_sm100.py)
- `PIPE` = /venv/main/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/pipeline/sm100.py (+ helpers.py)
- `DESC` = /venv/main/lib/python3.12/site-packages/flash_attn/cute/mma_sm100_desc.py (Python port of CUTLASS descriptor header)
- `BSL` = .../cutlass/utils/blockscaled_layout.py
- **PTXDUMP** = scratchpad/fa4_ptx_dump/*.sm_100a.ptx — REAL PTX of FA4 fwd compiled with
  use_2cta_instrs=True on this B200 (CUTE_DSL_KEEP=ptx; hd128, non-causal, seq 512).
  89 occurrences of `cta_group::2`.
- **PTXAS** = scratchpad/ptx_2cta_check.cu — every form below compiles clean with
  `nvcc -gencode=arch=compute_100a,code=sm_100a` (CUDA 13.0). Compile it again after edits to re-verify.
- Live probe = scratchpad/probe_2cta.py — prints the 2-CTA tiled-MMA A/B/C partitioning (run it, output quoted below).

**Bonus finding**: FA4's dispatch (interface.py:585-603) enables 2-CTA for hd128, non-causal,
non-split, seqlen_q > 2*tile_m on sm_100/110 — i.e. THE FA4 BASELINE YOU BENCH AGAINST IS
ALREADY A 2-CTA KERNEL at 4k-64k.

---

## Q1. tcgen05.mma cta_group::2, kind::mxf4nvf4, idesc M field, D address

ANSWER: Same syntax as the current 1-CTA instruction with `cta_group::1 -> cta_group::2`.
Issued by ONE elected thread in the LEADER (even-rank) CTA only (PTXDUMP shows
`@leader_thread tcgen05.mma.cta_group::2...` inside `if is_leader_cta`). The idesc M field is
`m_dim = M >> 4`, a 5-bit field at bits 24-28 (DESC:141,158), so M=256 -> 16<<24 = 1<<28.
Your existing `(M>>7)<<27` formula produces exactly the same bits for M in {128,256} — keep it,
just pass M=256. FA4's QK idesc constant in PTXDUMP is `0x10200490`: bits24-28 = 16 (M=256 ✓),
bits17-22 = 16 (N=128 ✓). All tmem operands (D, A-from-tmem, SFA, SFB) are per-CTA-local
addresses interpreted IN EACH CTA of the pair: each CTA's tmem gets its own 128 rows of D
(FA4 softmax/correction/epilogue code is identical between 1-CTA and 2-CTA for tmem access).

EXACT PTX (ptxas-validated on sm_100a, both operand forms):
```
tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [d_tmem],  a_desc,  b_desc, idesc, [sfa_tmem], [sfb_tmem], p;
tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [d_tmem], [a_tmem], b_desc, idesc, [sfa_tmem], [sfb_tmem], p;
```
EVIDENCE: DESC:141 `m_dim = M >> 4  # 5-bit field`; DESC:158 `desc |= (m_dim & 0x1F) << 24`;
PTXDUMP `tcgen05.mma.cta_group::2.kind::f16 [tmem_acc], fa_fwd_q_smem_desc_0, smem_desc_b_0, fa_fwd_qk_mma_idesc(=0x10200490), 0;`
and PV form `[tmem_acc], [tmem_a + 0x8], smem_desc_b, idesc, 1`. FA4SRC:177-179 "With 2CTA, the
MMA tiler M covers both CTAs". mma.py:581 allows M in {128,256} for CtaGroup.TWO (N%16==0 constraint, mma.py:583).

## Q2. B-operand sourcing: SPLIT along N, contiguous halves, same smem offset in both CTAs

ANSWER: FA4 SPLITS K and V across the pair (scheme a), NOT multicast. Each CTA holds HALF of
the B operand along the N dimension at the SAME local smem offset: CTA rank r holds
B rows [r*N/2, (r+1)*N/2) x full K. For QK (B=K, N=128 kv rows): CTA0 holds kv rows 0-63,
CTA1 holds 64-127. For PV (B=V^T, N=128 head-dim): CTA0 holds head-dim rows 0-63, CTA1 64-127,
for the full kv extent of the tile. The single leader-issued MMA's b_desc is a LOCAL smem
address; hardware reads each CTA's half via the pair path. KV smem per CTA HALVES.

Live probe output (M=256, N=128, K=128, CtaGroup.TWO — QK and PV identical):
```
thr_id qk: 2:1
QK partition_shape_A (M,K): ((128, 16), 1, 8)     # A split along M: own 128 rows
QK partition_shape_B (N,K): ((64, 16), 1, 8)      # B split along N: 64 rows per CTA
QK partition_shape_C (M,N): ((128, 128), 1, 1)    # C per CTA: own 128 rows x FULL N
QK _thrfrg_B: ((2,(1,1)),((64,16),(1,8))):((64@0,...))   # peer stride 64@0 -> contiguous halves
```
EVIDENCE: FA4SRC:391 `smem_size_kv_per_stage = max(...) // self.cta_group_size`;
FA4SRC:1463 `tSgK = thr_mma_qk.partition_B(gK)` (per-CTA slice); probe above.

**CRITICAL design consequence: KV MULTICAST (plan scheme b) IS INCOMPATIBLE WITH THE
cta_group::2 MMA.** Multicast writes IDENTICAL data at the same offset in both CTAs, but the
2-CTA MMA needs DIFFERENT N-halves at the same offset. Multicast KV only works while MMAs are
still per-CTA cta_group::1 (plan step 2 correctness gate). For step 3+ (M=256 QK), you must
switch the loads to split halves (each CTA TMAs its own 64 kv rows / 64 head-dim rows).
Suggested amendment: make step 2 load SPLIT halves into both CTAs but run per-CTA MMAs on the
half tiles... simpler: skip multicast entirely and go straight from scaffold to split+M=256.

## Q3. SFA / SFB placement for block-scaled cta_group::2

ANSWER: SFA (=Q sf for QK, P sf for PV): per-CTA — each CTA's tmem holds the scale factors for
ITS OWN 128 A rows (your existing per-CTA SFQ/SFP layout is already correct, unchanged).
SFB (=K sf, V sf): DUPLICATED — each CTA's smem (and thus tmem after the cp) holds the sf for
the FULL N (all 128 kv rows / all 128 head-dim), because each CTA's tensor core computes all N
columns for its M-half. Both tmem addresses in the instruction are per-CTA-local.
Consequence for you: even with split K/V data, load the FULL k/v sf atoms into BOTH CTAs' smem
(1KB per tile — cheap) and keep the per-CTA tcgen05.cp into own tmem.

EVIDENCE: BSL:162-165 (make_smem_layout_sfa) `sfa_tile_shape = (mma_tiler[0] // size(thr_id), K)`
= M/2 per CTA; BSL:231-234 (make_smem_layout_sfb) `sfb_tile_shape = (round_up(mma_tiler[1],128), K)`
= FULL N per CTA (not divided by cta_group). make_tmem_layout_sfa/sfb divide M by atom_thr_size
but never N (BSL:471-516).

## Q4. tcgen05.commit multicast

ANSWER: The leader's MMA warp (one elected thread) commits with a `.multicast::cluster`
qualifier and a 16-bit CTA mask (bit i = cluster rank i; mask=3 for both CTAs of the pair).
On completion of the previously issued cta_group::2 MMAs, the mbarrier at the given LOCAL
OFFSET is arrived in EVERY CTA named by the mask — this is how one commit wakes softmax in
both CTAs. Only the leader executes it (it issued the MMAs).

EXACT PTX (ptxas-validated; identical form in PTXDUMP with mask %rs16=3):
```
tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [mbar_local_offset], mask16;
tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [mbar];   // non-multicast variant
```
EVIDENCE: PTXDUMP `mov.b16 %rs16, 3; tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%r5], %rs16;`
helpers.py:374-396 arrive_tcgen05mma(mask, cta_group) under elect_one; PipelineUmmaAsync producer
mask = `make_layout_image_mask(... mode=0)` = both ranks (sm100.py:594-613).

## Q5. Remote mbarrier arrive (follower -> leader)

ANSWER: Convert the local mbarrier address to the peer CTA's DSMEM address with `mapa`, then
arrive on it. This is how softmax threads in BOTH CTAs signal the leader's P-ready barriers
(PipelineAsyncUmma producer arrives on `_compute_leading_cta_rank` = rank & ~1). The DSL emits
the unqualified form; CUTLASS C++ uses the explicit `.release.cluster` form — both are
ptxas-valid; prefer the explicit one.

EXACT PTX (both validated; second form is what FA4 actually emits):
```
mapa.shared::cluster.u32  %r_remote, %r_local_mbar, %r_peer_rank;
mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%r_remote];        // explicit (recommended)
mbarrier.arrive.shared::cluster.b64 _, [%r_remote], 1;                     // as emitted by DSL
```
Remote expect_tx also validated: `mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64 _, [%r_remote], bytes;`
EVIDENCE: PTXDUMP `mapa.shared::cluster.u32 %r617, %r615, %r616; ... mbarrier.arrive.shared::cluster.b64 _, [%r617], %r618;`
(11 occurrences); mbar.py:348-387 (mapa when peer_cta_rank given).

## Q6. tcgen05.alloc / dealloc / relinquish under cta_group::2

ANSWER: BOTH CTAs execute all three with the `.cta_group::2` forms (FA4's allocator warp is not
leader-gated). Column accounting is UNCHANGED per CTA: FA4 allocates the full 512 columns in
each CTA; each CTA's tmem is its own 512 cols; the same per-CTA tmem offsets are used in both.
Dealloc requires a cross-CTA handshake first so neither CTA frees while the peer might still
run MMAs that write both CTAs' tmem: each CTA's allocator warp does a REMOTE arrive on the
PEER's dealloc mbarrier (rank^1), waits its OWN barrier phase 0, then deallocs.

EXACT PTX (validated; identical in PTXDUMP):
```
tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [smem_dst], ncols;
tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;
tcgen05.dealloc.cta_group::2.sync.aligned.b32 taddr, ncols;
```
EVIDENCE: PTXDUMP all three with cta_group::2; tmem_allocator.py free(): `mbarrier_arrive(dealloc_mbar, rank^1); mbarrier_wait(dealloc_mbar, 0); dealloc_tmem(is_two_cta=True)`.
Note: per-CTA tcgen05.ld/st/cp (cta_group::1 semantics) on your own tmem remain legal and
unchanged — FA4's softmax does exactly that in 2-CTA mode.

## Q7. TMA forms: cta_group::2 and multicast

ANSWER (cta_group::2 — the one FA4 uses for Q/K/V): each CTA issues its own slice's TMA with
`.cta_group::2` appended; the mbarrier operand is a LOCAL offset but the completion (tx bytes)
is signaled on the mbarrier at that offset in the LEADER (even-rank) CTA of the pair. Only the
leader performs `mbarrier.arrive.expect_tx` (PipelineTmaUmma.producer_acquire is is_leader_cta-
gated) and the expected bytes are the FULL pair total (`tma_copy_bytes *= cta_group_size`).
ANSWER (multicast — NOT used by FA4 2-CTA fwd): mask16 bit i = receiving cluster rank; the SAME
data lands at the same offset in every masked CTA and EACH receiving CTA's own mbarrier at the
given offset receives the tx count (each recipient does its own expect_tx).

EXACT PTX (first is FA4's emitted form; all ptxas-validated):
```
cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2
    [dst_smem], [tmap, {x, y}], [mbar_local_offset_signals_LEADER], cache_policy;
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint
    [dst_smem], [tmap, {x, y}], [mbar], ctaMask16, cache_policy;
```
CONSTRAINT (bit us in validation): plain NON-tensor `cp.async.bulk` REJECTS `.cta_group::2`
("Illegal modifier"). For the sf-atom loads you have two working options, both validated:
  (a) remote-mbar form: `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [dst_local],[src],size,[mapa'd_remote_mbar];`
  (b) complete sf loads on a CTA-LOCAL barrier with a local expect_tx (each CTA needs the sf in
      its own smem anyway for SFB duplication — see Q3).
EVIDENCE: PTXDUMP 4d cta_group::2 loads all signaling `[%r19]`-style local offsets while
producer_acquire expect_tx is leader-only with doubled bytes (FA4SRC:629, PIPE sm100.py:354-372);
ptxas error log for the plain-bulk cta_group attempt.

## Q8. Cluster launch, sync, rank, mbarrier fencing

ANSWER: Launch with `__cluster_dims__(2,1,1)` on the kernel (ptxas-validated) or
cudaLaunchKernelEx + cudaLaunchAttributeClusterDimension = {2,1,1}; grid.x must be a multiple
of 2; persistent grid = 2*floor(sm_count/2). Rank = `%cluster_ctarank` (0 = leader of the pair
when cluster is (2,1,1)). Cluster-wide sync = barrier.cluster.arrive(.relaxed) + wait — required
ONCE after mbarrier init (with the fence) before any cross-CTA mbarrier use, and again before
tmem dealloc / kernel exit (the dealloc handshake in Q6 covers the latter). Keep the existing
`fence.mbarrier_init.release.cluster`.

EXACT PTX (validated; DUMP emits without .aligned — both legal):
```
mov.u32 %r, %cluster_ctarank;
barrier.cluster.arrive.relaxed.aligned;      // or barrier.cluster.arrive (full release)
barrier.cluster.wait.aligned;
fence.mbarrier_init.release.cluster;
```
EVIDENCE: PTXDUMP `barrier.cluster.arrive.relaxed` / `barrier.cluster.wait` /
`fence.mbarrier_init.release.cluster`; pipeline create() does mbarrier_init_fence +
agent_sync(ThreadBlockCluster, is_relaxed=True) (PIPE sm100.py:305-310).

## Q9. How the leader observes the follower's KV TMA completion

ANSWER: Automatically, via Q7: both CTAs' half-tile TMAs carry `.cta_group::2` and complete-tx
on the LEADER's KVFull mbarrier (local offset, leader-routed). The leader expect_tx's the sum
of both halves (+ sf bytes if you route those there too, see Q7 options). The follower's load
warp never touches the leader's barrier explicitly and the leader's MMA warp waits only on its
own local KVFull, exactly like today. Data written to the FOLLOWER's smem is visible to the
leader-issued MMA once the leader's barrier trips (TMA completion semantics cover the pair via
the async proxy; the existing tcgen05.fence::after pattern stays as-is).
KVEmpty release: the leader's MMA commits with multicast mask 3 (Q4) so EACH CTA's local
KVEmpty barrier fires and each CTA's load warp waits locally, unchanged.

EVIDENCE: chain of FA4SRC:391 (halved smem) + FA4SRC:629 (doubled tx bytes) + PIPE
producer_acquire leader-only expect_tx + load_KV passing `pipeline_kv.producer_get_barrier`
(local) + atom cta_group=TWO (FA4SRC:632) + PTXDUMP cta_group::2 loads. Flag: "completion lands
on the pair leader's barrier" is the only reading consistent with all of the above, but it is
inferred, not stated in a doc I could read — verify once on device with a 2-line smoke test
(follower-issued TMA, leader waits) before building on it.

## Q10. How the follower's softmax knows S is ready

ANSWER: Via the leader's commit MULTICAST (Q4). SFull is a UMMA->AsyncThread pipeline
(PipelineUmmaAsync): the leader's MMA warp, after issuing the M=256 QK, executes one
tcgen05.commit with multicast mask = both ranks; when the MMA completes, the SFull barrier at
that offset fires in BOTH CTAs. Each CTA's softmax waits on its LOCAL SFull with the usual
parity — no other change. The follower needs no extra wait before its S tmem is written: the
leader-issued MMA writes each CTA's own tmem half directly, and the commit-multicast IS the
readiness signal. Reverse direction (P0/P1 ready, SEmpty release by softmax): softmax lanes in
BOTH CTAs do the REMOTE arrive of Q5 onto the LEADER's barrier (counts double: 256/512 for your
P0/P1), because only the leader's MMA warp consumes them.

EVIDENCE: PipelineUmmaAsync.producer_commit -> arrive(index, producer_mask=_compute_tmem_sync_mask
(both ranks), cta_group) (PIPE sm100.py:737-750); consumer_release -> arrive(index,
consumer_mask=_compute_peer_cta_rank()=leader rank) = remote mbarrier arrive (helpers.py:321-359);
FA4SRC:971-981 "the non-MMA side spans both CTAs in the cluster" (cluster consumer groups with
len * cta_group_size arrivals).

---

## Residual uncertainties (flagged)

1. Q9's leader-routing of cta_group::2 TMA completions: inferred (airtight chain, but no direct
   doc quote). One-kernel device smoke test recommended before depending on it.
2. tcgen05.cp.cta_group::2 semantics (leader-issued SF copy writing BOTH CTAs' tmem): syntax
   validates, but I did not find a usage to confirm semantics. AVOID: keep per-CTA
   `tcgen05.cp.cta_group::1` into own tmem (proven legal in 2-CTA kernels). If the follower's
   own MMA warp is parked, the follower still needs SFB in its tmem -> have the follower's
   (otherwise idle) warp 12 do its own sf cps, then signal the leader via
   `tcgen05.commit.cta_group::1...shared::cluster.b64 [mapa'd leader SFReady mbar]` (the commit
   mbar operand is already shared::cluster space in your existing helper).
3. The `.multicast::cluster` "each recipient's own barrier gets tx" semantics: standard Hopper
   behavior, not re-verified on Blackwell here (FA4 2-CTA doesn't use multicast — and see Q2:
   multicast is incompatible with the M=256 MMA anyway).

---

## DEVICE SMOKE-TEST RESULTS (2026-07-04, smoke_2cta.cu on this B200)

Both flagged uncertainties RESOLVED, with one CORRECTION to Q7/Q9:

- Q7/Q9 CORRECTION: a plain local mbarrier offset does NOT auto-route to the
  leader (that version deadlocks).  The mbarrier operand of a cta_group::2
  TMA must be the PAIR LEADER's barrier as an explicit shared::cluster
  address: `mapa.shared::cluster.u32 mb, local_mbar, (rank & ~1)`.  (FA4's
  PTX does the same thing by clearing the rank bit: `and.b32 %r33, addr,
  ~0x1000000`.)  With that form: both CTAs issue their own half-box TMA
  (dst = own local smem, per-rank gmem coords), leader alone does
  expect_tx(pair total) and waits its local barrier.  PASS.
- Q4 + uncertainty #2: leader-issued `tcgen05.cp.cta_group::2` DOES read
  each CTA's own smem at the given offset and write each CTA's own tmem
  (verified with per-CTA patterns), and
  `tcgen05.commit.cta_group::2...multicast::cluster` (mask 3) fires the
  local barrier in BOTH CTAs.  PASS.  => no follower sf-relay warp needed;
  the leader's mma warp cps sf for the whole pair.
