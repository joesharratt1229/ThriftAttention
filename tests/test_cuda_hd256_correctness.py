
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")
_C = pytest.importorskip("thriftattention._C")


CONTEXT_LENGTHS = (4096, 8192, 32768, 131072)
CONTEXT_LENGTHS_DECODE = (512, 1024, 4096, 8192, 32768, 131072)
DTYPES = (torch.float16, torch.bfloat16)
BATCH_SIZES = (1,)


def _requires_sm120_cuda() -> None:
	if not torch.cuda.is_available():
		pytest.skip("CUDA device required")
	if torch.cuda.get_device_capability() < (12, 0):
		pytest.skip("SM120 CUDA device required")


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
	return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _nvfp4_quantize_qkv(
	q: torch.Tensor,
	k: torch.Tensor,
	v: torch.Tensor,
	*,
	is_bf16: bool,
	permute_k: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	q_packed, q_scale = _C.nvfp4_quantize(q, is_bf16)
	if permute_k:
		k_packed, k_scale = _C.nvfp4_quantize_permuted(k, is_bf16)
	else:
		k_packed, k_scale = _C.nvfp4_quantize(k, is_bf16)
	v_packed_t, v_scale_t = _C.nvfp4_quantize_transposed(v, is_bf16)
	return q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_len", CONTEXT_LENGTHS_DECODE)
@pytest.mark.parametrize("batch", BATCH_SIZES)
def test_single_query_nvfp4_hd256_matches_sdpa(dtype: torch.dtype, kv_len: int, batch: int) -> None:
	_requires_sm120_cuda()
	torch.manual_seed(10)
	device = torch.device("cuda")
	is_bf16 = dtype == torch.bfloat16
	q_heads, kv_heads, head_dim = 4, 2, 256
	groups = q_heads // kv_heads

	q = (torch.randn(batch, q_heads, 1, head_dim, device=device, dtype=dtype)).contiguous()
	k = (torch.randn(batch, kv_heads, kv_len, head_dim, device=device, dtype=dtype)).contiguous()
	v = (torch.randn(batch, kv_heads, kv_len, head_dim, device=device, dtype=dtype)).contiguous()
	q_grouped = q.reshape(batch, kv_heads, groups, head_dim).contiguous()

	packed = _nvfp4_quantize_qkv(q_grouped, k, v, is_bf16=is_bf16, permute_k=False)
	fp4_out = _C.fp4_attention_single_query_nvfp4_packed(*packed, is_bf16)
	fp4_out = fp4_out.reshape(batch, q_heads, 1, head_dim)

	num_kv_blocks = kv_len // 64
	k_mean = k.reshape(batch, kv_heads, num_kv_blocks, 64, head_dim).float().mean(dim=3).to(dtype)
	selected = _C.single_query_key_mean_topk(
		q_grouped,
		k_mean.contiguous(),
		num_kv_blocks,
		num_kv_blocks,
		is_bf16,
	)
	thrift_out = _C.thrift_attention_single_query_nvfp4_packed(
		q_grouped,
		k,
		v,
		selected,
		*packed,
		is_bf16,
	).reshape(batch, q_heads, 1, head_dim)

	k_ref = k.repeat_interleave(groups, dim=1)
	v_ref = v.repeat_interleave(groups, dim=1)
	ref = F.scaled_dot_product_attention(q.float(), k_ref.float(), v_ref.float(), is_causal=False)

	torch.cuda.synchronize()
	assert fp4_out.dtype == dtype
	assert thrift_out.dtype == dtype
	assert _cosine(thrift_out, ref) > 0.95
	assert _cosine(fp4_out, ref) > 0.95
	


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_len", CONTEXT_LENGTHS)
@pytest.mark.parametrize("batch", BATCH_SIZES)
def test_tiled_nvfp4_hd256_matches_sdpa(dtype: torch.dtype, kv_len: int, batch: int) -> None:
	_requires_sm120_cuda()
	torch.manual_seed(11)
	device = torch.device("cuda")
	is_bf16 = dtype == torch.bfloat16
	q_heads, kv_heads, seq_len, head_dim = 2, 1, kv_len, 256
	groups = q_heads // kv_heads

	q = (torch.randn(batch, q_heads, seq_len, head_dim, device=device, dtype=dtype)).contiguous()
	k = (torch.randn(batch, kv_heads, seq_len, head_dim, device=device, dtype=dtype)).contiguous()
	v = (torch.randn(batch, kv_heads, seq_len, head_dim, device=device, dtype=dtype)).contiguous()

	packed = _nvfp4_quantize_qkv(q, k, v, is_bf16=is_bf16)
	fp4_out = _C.fp4_attention_causal_nvfp4_packed(*packed, is_bf16)

	num_q_blocks = seq_len // 128
	num_kv_blocks = seq_len // 128
	selected = (
		torch.arange(num_kv_blocks, device=device, dtype=torch.int32)
		.view(1, 1, num_kv_blocks)
		.expand(batch * q_heads, num_q_blocks, num_kv_blocks)
		.contiguous()
	)
	thrift_out = _C.thrift_attention_causal_nvfp4_packed(q, k, v, selected, *packed, is_bf16)

	k_ref = k.repeat_interleave(groups, dim=1)
	v_ref = v.repeat_interleave(groups, dim=1)
	ref = F.scaled_dot_product_attention(q.float(), k_ref.float(), v_ref.float(), is_causal=True)

	torch.cuda.synchronize()
	assert fp4_out.dtype == dtype
	assert thrift_out.dtype == dtype
	assert _cosine(thrift_out, ref) > 0.95
	assert _cosine(fp4_out, ref) > 0.95
	

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_len", CONTEXT_LENGTHS)
@pytest.mark.parametrize("batch", BATCH_SIZES)
def test_tiled_nvfp4_hd256_noncausal_matches_sdpa(dtype: torch.dtype, kv_len: int, batch: int) -> None:
	_requires_sm120_cuda()
	torch.manual_seed(20)
	device = torch.device("cuda")
	is_bf16 = dtype == torch.bfloat16
	q_heads, kv_heads, seq_len, head_dim = 2, 1, kv_len, 256
	groups = q_heads // kv_heads

	q = (torch.randn(batch, q_heads, seq_len, head_dim, device=device, dtype=dtype)).contiguous()
	k = (torch.randn(batch, kv_heads, seq_len, head_dim, device=device, dtype=dtype)).contiguous()
	v = (torch.randn(batch, kv_heads, seq_len, head_dim, device=device, dtype=dtype)).contiguous()

	packed = _nvfp4_quantize_qkv(q, k, v, is_bf16=is_bf16)
	fp4_out = _C.fp4_attention_noncausal_nvfp4_packed(*packed, is_bf16)

	num_q_blocks = seq_len // 128
	num_kv_blocks = seq_len // 128
	selected = (
		torch.arange(num_kv_blocks, device=device, dtype=torch.int32)
		.view(1, 1, num_kv_blocks)
		.expand(batch * q_heads, num_q_blocks, num_kv_blocks)
		.contiguous()
	)
	thrift_out = _C.thrift_attention_noncausal_nvfp4_packed(q, k, v, selected, *packed, is_bf16)

	k_ref = k.repeat_interleave(groups, dim=1)
	v_ref = v.repeat_interleave(groups, dim=1)
	ref = F.scaled_dot_product_attention(q.float(), k_ref.float(), v_ref.float(), is_causal=False)

	torch.cuda.synchronize()
	assert fp4_out.dtype == dtype
	assert thrift_out.dtype == dtype
	assert _cosine(thrift_out, ref) > 0.95
	assert _cosine(fp4_out, ref) > 0.95
	

def _mxfp4_quantize_qkv(
	q: torch.Tensor,
	k: torch.Tensor,
	v: torch.Tensor,
	*,
	is_bf16: bool,
	permute_k: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	q_packed, q_scale = _C.mxfp4_quantize(q, is_bf16)
	if permute_k:
		k_packed, k_scale = _C.mxfp4_quantize_permuted(k, is_bf16)
	else:
		k_packed, k_scale = _C.mxfp4_quantize(k, is_bf16)
	v_packed_t, v_scale_t = _C.mxfp4_quantize_transposed(v, is_bf16)
	return q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_len", CONTEXT_LENGTHS_DECODE)
@pytest.mark.parametrize("batch", BATCH_SIZES)
def test_single_query_mxfp4_hd256_matches_sdpa(dtype: torch.dtype, kv_len: int, batch: int) -> None:
	_requires_sm120_cuda()
	torch.manual_seed(12)
	device = torch.device("cuda")
	is_bf16 = dtype == torch.bfloat16
	q_heads, kv_heads, head_dim = 4, 2, 256
	groups = q_heads // kv_heads

	q = (torch.randn(batch, q_heads, 1, head_dim, device=device, dtype=dtype)).contiguous()
	k = (torch.randn(batch, kv_heads, kv_len, head_dim, device=device, dtype=dtype)).contiguous()
	v = (torch.randn(batch, kv_heads, kv_len, head_dim, device=device, dtype=dtype)).contiguous()
	q_grouped = q.reshape(batch, kv_heads, groups, head_dim).contiguous()

	packed = _mxfp4_quantize_qkv(q_grouped, k, v, is_bf16=is_bf16, permute_k=False)
	fp4_out = _C.fp4_attention_single_query_mxfp4_packed(*packed, is_bf16)
	fp4_out = fp4_out.reshape(batch, q_heads, 1, head_dim)

	num_kv_blocks = kv_len // 64
	k_mean = k.reshape(batch, kv_heads, num_kv_blocks, 64, head_dim).float().mean(dim=3).to(dtype)
	selected = _C.single_query_key_mean_topk(
		q_grouped,
		k_mean.contiguous(),
		num_kv_blocks,
		num_kv_blocks,
		is_bf16,
	)
	thrift_out = _C.thrift_attention_single_query_mxfp4_packed(
		q_grouped,
		k,
		v,
		selected,
		*packed,
		is_bf16,
	).reshape(batch, q_heads, 1, head_dim)

	k_ref = k.repeat_interleave(groups, dim=1)
	v_ref = v.repeat_interleave(groups, dim=1)
	ref = F.scaled_dot_product_attention(q.float(), k_ref.float(), v_ref.float(), is_causal=False)

	torch.cuda.synchronize()
	assert thrift_out.dtype == dtype
	assert fp4_out.dtype == dtype
	assert _cosine(thrift_out, ref) > 0.95
	assert _cosine(fp4_out, ref) > 0.95
	


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_len", CONTEXT_LENGTHS)
@pytest.mark.parametrize("batch", BATCH_SIZES)
def test_tiled_mxfp4_hd256_matches_sdpa(dtype: torch.dtype, kv_len: int, batch: int) -> None:
	_requires_sm120_cuda()
	torch.manual_seed(13)
	device = torch.device("cuda")
	is_bf16 = dtype == torch.bfloat16
	q_heads, kv_heads, seq_len, head_dim = 2, 1, kv_len, 256
	groups = q_heads // kv_heads

	q = (torch.randn(batch, q_heads, seq_len, head_dim, device=device, dtype=dtype)).contiguous()
	k = (torch.randn(batch, kv_heads, seq_len, head_dim, device=device, dtype=dtype)).contiguous()
	v = (torch.randn(batch, kv_heads, seq_len, head_dim, device=device, dtype=dtype)).contiguous()

	packed = _mxfp4_quantize_qkv(q, k, v, is_bf16=is_bf16)
	fp4_out = _C.fp4_attention_causal_mxfp4_packed(*packed, is_bf16)

	num_q_blocks = seq_len // 128
	num_kv_blocks = seq_len // 128
	selected = (
		torch.arange(num_kv_blocks, device=device, dtype=torch.int32)
		.view(1, 1, num_kv_blocks)
		.expand(batch * q_heads, num_q_blocks, num_kv_blocks)
		.contiguous()
	)
	thrift_out = _C.thrift_attention_causal_mxfp4_packed(q, k, v, selected, *packed, is_bf16)

	k_ref = k.repeat_interleave(groups, dim=1)
	v_ref = v.repeat_interleave(groups, dim=1)
	ref = F.scaled_dot_product_attention(q.float(), k_ref.float(), v_ref.float(), is_causal=True)

	torch.cuda.synchronize()
	assert fp4_out.dtype == dtype
	assert thrift_out.dtype == dtype
	assert _cosine(fp4_out, ref) > 0.95
	assert _cosine(thrift_out, ref) > 0.95


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_len", CONTEXT_LENGTHS)
@pytest.mark.parametrize("batch", BATCH_SIZES)
def test_tiled_mxfp4_hd256_noncausal_matches_sdpa(dtype: torch.dtype, kv_len: int, batch: int) -> None:
	_requires_sm120_cuda()
	torch.manual_seed(21)
	device = torch.device("cuda")
	is_bf16 = dtype == torch.bfloat16
	q_heads, kv_heads, seq_len, head_dim = 2, 1, kv_len, 256
	groups = q_heads // kv_heads

	q = (torch.randn(batch, q_heads, seq_len, head_dim, device=device, dtype=dtype)).contiguous()
	k = (torch.randn(batch, kv_heads, seq_len, head_dim, device=device, dtype=dtype)).contiguous()
	v = (torch.randn(batch, kv_heads, seq_len, head_dim, device=device, dtype=dtype)).contiguous()

	packed = _mxfp4_quantize_qkv(q, k, v, is_bf16=is_bf16)
	fp4_out = _C.fp4_attention_noncausal_mxfp4_packed(*packed, is_bf16)

	num_q_blocks = seq_len // 128
	num_kv_blocks = seq_len // 128
	selected = (
		torch.arange(num_kv_blocks, device=device, dtype=torch.int32)
		.view(1, 1, num_kv_blocks)
		.expand(batch * q_heads, num_q_blocks, num_kv_blocks)
		.contiguous()
	)
	thrift_out = _C.thrift_attention_noncausal_mxfp4_packed(q, k, v, selected, *packed, is_bf16)

	k_ref = k.repeat_interleave(groups, dim=1)
	v_ref = v.repeat_interleave(groups, dim=1)
	ref = F.scaled_dot_product_attention(q.float(), k_ref.float(), v_ref.float(), is_causal=False)

	torch.cuda.synchronize()
	assert fp4_out.dtype == dtype
	assert thrift_out.dtype == dtype
	assert _cosine(fp4_out, ref) > 0.95
	assert _cosine(thrift_out, ref) > 0.95
