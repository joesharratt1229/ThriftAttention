
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")
_C = pytest.importorskip("thriftattention._C")


CONTEXT_LENGTHS = (4096, 8192, 32768)
DTYPES = (torch.float16, torch.bfloat16)
HEAD_DIMS = (64, 128)
BUDGETS = (0.00, 0.05, 0.10, 0.15)
BATCH_SIZES = (1,)


def _requires_sm120_cuda() -> None:
	if not torch.cuda.is_available():
		pytest.skip("CUDA device required")
	if torch.cuda.get_device_capability() < (12, 0):
		pytest.skip("SM120 CUDA device required")


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
	return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _block_size(head_dim: int) -> int:
	return 128 if head_dim == 256 else 64


def _nvfp4_quantize_qkv(
	q: torch.Tensor,
	k: torch.Tensor,
	v: torch.Tensor,
	*,
	is_bf16: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	q_packed, q_scale = _C.nvfp4_quantize(q, is_bf16)
	k_packed, k_scale = _C.nvfp4_quantize_permuted(k, is_bf16)
	v_packed_t, v_scale_t = _C.nvfp4_quantize_transposed(v, is_bf16)
	return q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t


def _budget_topk_selection(
	q: torch.Tensor,
	k: torch.Tensor,
	budget: float,
	*,
	causal: bool,
	is_bf16: bool,
) -> torch.Tensor:
	batch, q_heads, seq_len, head_dim = q.shape
	kv_heads = k.shape[1]
	groups = q_heads // kv_heads
	block = _block_size(head_dim)
	num_q_blocks = seq_len // block
	num_kv_blocks = k.shape[2] // block
	topk_count = max(1, round(budget * num_kv_blocks))

	q_mean = (
		q.reshape(batch, q_heads, num_q_blocks, block, head_dim)
		.float()
		.mean(dim=3)
		.to(q.dtype)
		.contiguous()
	)
	k_mean = (
		k.reshape(batch, kv_heads, num_kv_blocks, block, head_dim)
		.float()
		.mean(dim=3)
		.to(k.dtype)
		.repeat_interleave(groups, dim=1)
		.contiguous()
	)
	selected = _C.block_mean_topk(q_mean, k_mean, topk_count, causal, is_bf16)
	return selected.reshape(batch * q_heads, num_q_blocks, topk_count).contiguous()


def _run_budget_case(
	dtype: torch.dtype,
	kv_len: int,
	batch: int,
	head_dim: int,
	budget: float,
	*,
	causal: bool,
	seed: int,
) -> tuple[float, float]:
	"""Returns (thrift cosine vs SDPA, pure-FP4 cosine vs SDPA)."""
	torch.manual_seed(seed)
	device = torch.device("cuda")
	is_bf16 = dtype == torch.bfloat16
	q_heads, kv_heads, seq_len = 2, 1, kv_len
	groups = q_heads // kv_heads

	q = torch.randn(batch, q_heads, seq_len, head_dim, device=device, dtype=dtype).contiguous()
	k = torch.randn(batch, kv_heads, seq_len, head_dim, device=device, dtype=dtype).contiguous()
	v = torch.randn(batch, kv_heads, seq_len, head_dim, device=device, dtype=dtype).contiguous()

	packed = _nvfp4_quantize_qkv(q, k, v, is_bf16=is_bf16)
	if causal:
		fp4_out = _C.fp4_attention_causal_nvfp4_packed(*packed, is_bf16)
	else:
		fp4_out = _C.fp4_attention_noncausal_nvfp4_packed(*packed, is_bf16)

	selected = _budget_topk_selection(q, k, budget, causal=causal, is_bf16=is_bf16)
	if causal:
		thrift_out = _C.thrift_attention_causal_nvfp4_packed(q, k, v, selected, *packed, is_bf16)
	else:
		thrift_out = _C.thrift_attention_noncausal_nvfp4_packed(q, k, v, selected, *packed, is_bf16)

	k_ref = k.repeat_interleave(groups, dim=1)
	v_ref = v.repeat_interleave(groups, dim=1)
	ref = F.scaled_dot_product_attention(q.float(), k_ref.float(), v_ref.float(), is_causal=causal)

	torch.cuda.synchronize()
	assert thrift_out.dtype == dtype
	assert fp4_out.dtype == dtype
	return _cosine(thrift_out, ref), _cosine(fp4_out, ref)


@pytest.mark.parametrize("budget", BUDGETS)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_len", CONTEXT_LENGTHS)
@pytest.mark.parametrize("batch", BATCH_SIZES)
def test_tiled_nvfp4_budget_causal_matches_sdpa(
	dtype: torch.dtype, kv_len: int, batch: int, head_dim: int, budget: float
) -> None:
	_requires_sm120_cuda()
	thrift_cos, fp4_cos = _run_budget_case(
		dtype, kv_len, batch, head_dim, budget, causal=True, seed=30
	)
	assert thrift_cos > 0.98
	# Upgrading the selected blocks to FP16 must not degrade the pure-FP4 baseline.
	assert thrift_cos >= fp4_cos - 0.01


@pytest.mark.parametrize("budget", BUDGETS)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_len", CONTEXT_LENGTHS)
@pytest.mark.parametrize("batch", BATCH_SIZES)
def test_tiled_nvfp4_budget_noncausal_matches_sdpa(
	dtype: torch.dtype, kv_len: int, batch: int, head_dim: int, budget: float
) -> None:
	_requires_sm120_cuda()
	thrift_cos, fp4_cos = _run_budget_case(
		dtype, kv_len, batch, head_dim, budget, causal=False, seed=31
	)
	assert thrift_cos > 0.95
	assert thrift_cos >= fp4_cos - 0.01


@pytest.mark.parametrize("causal", (True, False))
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_tiled_nvfp4_budget_monotonic(dtype: torch.dtype, head_dim: int, causal: bool) -> None:
	_requires_sm120_cuda()
	cosines = [
		_run_budget_case(dtype, 8192, 1, head_dim, budget, causal=causal, seed=32)[0]
		for budget in BUDGETS
	]
	for smaller, larger in zip(cosines, cosines[1:]):
		assert larger >= smaller - 0.005, f"budget increase degraded cosine: {cosines}"
