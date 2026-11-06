#include <torch/extension.h>

#include <algorithm>
#include <cstdlib>
#include <vector>

#include <cuda_runtime_api.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include "thriftattention/sm120/paged_bf16_prefix.h"

namespace {

void check_hi_qkv(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "Q/K/V must be CUDA tensors");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "Q/K/V must be 4D [batch, heads, seq, head_dim]");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0),
                "Q/K/V batch dimensions must match");
    TORCH_CHECK(k.size(1) == v.size(1), "K and V head dimensions must match");
    TORCH_CHECK(k.size(2) == v.size(2), "K and V sequence lengths must match");
    TORCH_CHECK(q.size(3) == k.size(3) && q.size(3) == v.size(3),
                "Q/K/V head dimensions must match");
}

void check_packed_qkv(
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t) {
    TORCH_CHECK(q_packed.is_cuda() && k_packed.is_cuda() && v_packed_t.is_cuda(),
                "packed Q/K/V must be CUDA tensors");
    TORCH_CHECK(q_scale.is_cuda() && k_scale.is_cuda() && v_scale_t.is_cuda(),
                "Q/K/V scales must be CUDA tensors");
    TORCH_CHECK(q_packed.dim() == 4 && k_packed.dim() == 4 && v_packed_t.dim() == 4,
                "packed Q/K/V must be 4D tensors");
    TORCH_CHECK(q_packed.size(0) == k_packed.size(0), "packed Q/K batch dimensions must match");
    TORCH_CHECK(q_packed.size(1) % k_packed.size(1) == 0,
                "Q heads must be divisible by KV heads");
    TORCH_CHECK(q_packed.size(3) == k_packed.size(3), "packed Q/K head dimensions must match");
}

// The packed inputs of the varlen kernels: every sequence starts at a multiple of 64 tokens of one batch row, and
// Q, K and V hold the same tokens (V padded to a multiple of 128 by nvfp4_quantize_transposed) on one device.
void check_varlen_inputs(
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    const at::Tensor& cu_seqlens,
    const at::Tensor& cu_seqlens_padded) {
    check_packed_qkv(q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t);
    for (const at::Tensor* t : {&k_packed, &v_packed_t, &q_scale, &k_scale, &v_scale_t, &cu_seqlens,
                                &cu_seqlens_padded}) {
        TORCH_CHECK(t->device() == q_packed.device(), "all varlen inputs must be on the device of q_packed");
    }
    for (const at::Tensor* t : {&q_packed, &k_packed, &v_packed_t}) {
        TORCH_CHECK(t->scalar_type() == at::kByte && t->is_contiguous(), "packed Q/K/V must be contiguous uint8");
    }
    for (const at::Tensor* t : {&q_scale, &k_scale, &v_scale_t}) {
        TORCH_CHECK(t->scalar_type() == at::kFloat8_e4m3fn && t->is_contiguous(),
                    "Q/K/V scales must be contiguous float8_e4m3fn");
    }
    const int64_t heads = q_packed.size(1);
    const int64_t kv_heads = k_packed.size(1);
    const int64_t tokens = q_packed.size(2);
    const int64_t head_dim = q_packed.size(3) * 2;
    const int64_t v_tokens = v_packed_t.size(3) * 2;
    TORCH_CHECK(q_packed.size(0) == 1, "varlen inputs hold all sequences in one batch row");
    TORCH_CHECK(k_packed.size(2) == tokens && tokens % 64 == 0 && v_tokens >= tokens,
                "packed Q, K and V must hold the same tokens, a multiple of 64");
    TORCH_CHECK(v_packed_t.size(1) == kv_heads && v_packed_t.size(2) == head_dim,
                "packed V must be [1, num_kv_heads, head_dim, v_tokens / 2]");
    TORCH_CHECK(q_scale.sizes() == at::IntArrayRef({1, heads, tokens, head_dim / 16}) &&
                k_scale.sizes() == at::IntArrayRef({1, kv_heads, tokens, head_dim / 16}) &&
                v_scale_t.sizes() == at::IntArrayRef({1, kv_heads, head_dim, v_tokens / 16}),
                "Q/K/V scales do not match the packed tensors");
    TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt && cu_seqlens_padded.scalar_type() == at::kInt &&
                cu_seqlens.dim() == 1 && cu_seqlens.is_contiguous() && cu_seqlens_padded.is_contiguous(),
                "cu_seqlens and cu_seqlens_padded must be contiguous 1-D int32 tensors");
    TORCH_CHECK(cu_seqlens.numel() == cu_seqlens_padded.numel() && cu_seqlens.numel() >= 1,
                "cu_seqlens and cu_seqlens_padded must both have num_seqs + 1 entries");
}

// A caller's buffer that a kernel writes in place: its device, dtype, rows and size, contiguous and 16-byte aligned
// (the epilogues store whole words).
void check_output_buffer(const at::Tensor& buffer, const at::Tensor& like, at::ScalarType dtype, int64_t rows,
                         int64_t numel, const char* name) {
    TORCH_CHECK(buffer.device() == like.device() && buffer.scalar_type() == dtype && buffer.dim() >= 1 &&
                buffer.size(0) == rows && buffer.numel() == numel && buffer.is_contiguous() &&
                reinterpret_cast<uintptr_t>(buffer.data_ptr()) % 16 == 0,
                name, " must be a contiguous, 16-byte aligned ", c10::toString(dtype), " tensor with ", rows,
                " rows and ", numel, " elements on the device of q_packed");
}

int single_query_mixed_target_split_ctas(int total_kv_blocks) {
    if (const char* env = std::getenv("SAGE_MIXED_SPLIT_CTAS")) {
        const int value = std::atoi(env);
        if (value > 0) {
            return value;
        }
    }
    return (total_kv_blocks <= 512) ? 384
         : (total_kv_blocks <= 1024) ? 384
         : 896;
}

}  // namespace

// Implemented in csrc/cuda/sm120/nvfp4/fp4_attention.cu.
void fp4_attention_causal_nvfp4(
    const void* Q, const void* K, const void* V,
    const void* S_Q, const void* S_K, const void* S_V,
    void* O, int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads, int head_dim, bool is_bf16);
void fp4_attention_noncausal_nvfp4(
    const void* Q, const void* K, const void* V,
    const void* S_Q, const void* S_K, const void* S_V,
    void* O, int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads, int head_dim, bool is_bf16);
void fp4_attention_varlen_nvfp4(
    const void* Q, const void* K, const void* V,
    const void* S_Q, const void* S_K, const void* S_V,
    const float* key_bias, void* O, const int* cu_seqlens, const int* cu_seqlens_padded,
    int num_seqs, int max_seqlen, int q_tokens, int k_tokens, int v_tokens,
    int num_q_heads, int num_kv_heads, int head_dim, bool causal, bool is_bf16,
    uint8_t* O_fp4, uint8_t* O_sf, const float* out_global_scale,
    const int* q_start, const int* out_start, float* lse_out, int out_tokens, cudaStream_t stream,
    const PagedBf16PrefixArgs* paged_prefix = nullptr);
// Implemented in csrc/cuda/sm120/mxfp4/fp4_attention.cu.
void fp4_attention_causal_mxfp4(
    const void* Q, const void* K, const void* V,
    const void* S_Q, const void* S_K, const void* S_V,
    void* O, int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads, int head_dim, bool is_bf16);
void fp4_attention_noncausal_mxfp4(
    const void* Q, const void* K, const void* V,
    const void* S_Q, const void* S_K, const void* S_V,
    void* O, int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads, int head_dim, bool is_bf16);
// Implemented in csrc/cuda/sm120/nvfp4/single_query_fp4_attention.cu.
void fp4_attention_single_query_nvfp4(
    const void* Q, const void* K, const void* V,
    const void* S_Q, const void* S_K, const void* S_V,
    void* O, void* workspace,
    int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads, int head_dim, bool is_bf16);
// Implemented in csrc/cuda/sm120/mxfp4/single_query_fp4_attention.cu.
void fp4_attention_single_query_mxfp4(
    const void* Q, const void* K, const void* V,
    const void* S_Q, const void* S_K, const void* S_V,
    void* O, void* workspace,
    int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads, int head_dim, bool is_bf16);

// Implemented in csrc/cuda/sm120/nvfp4/thrift_attention.cu.
void thrift_attention_causal_nvfp4(
    const void* Q_fp16, const void* K_fp16, const void* V_fp16,
    const void* selected_blocks, int topk_count,
    const void* topk_mask, int topk_word_count,
    const void* Q, const void* K, const void* V,
    const void* S_Q, const void* S_K, const void* S_V,
    void* O, void* rowmax_state, void* rowsum_state,
    int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads, int head_dim, bool is_bf16);
void thrift_attention_noncausal_nvfp4(
    const void* Q_fp16, const void* K_fp16, const void* V_fp16,
    const void* selected_blocks, int topk_count,
    const void* topk_mask, int topk_word_count,
    const void* Q, const void* K, const void* V,
    const void* S_Q, const void* S_K, const void* S_V,
    void* O, void* rowmax_state, void* rowsum_state,
    int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads, int head_dim, bool is_bf16);
// Implemented in csrc/cuda/sm120/mxfp4/thrift_attention.cu.
void thrift_attention_causal_mxfp4(
    const void* Q_fp16, const void* K_fp16, const void* V_fp16,
    const void* selected_blocks, int topk_count,
    const void* topk_mask, int topk_word_count,
    const void* Q, const void* K, const void* V,
    const void* S_Q, const void* S_K, const void* S_V,
    void* O, void* rowmax_state, void* rowsum_state,
    int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads, int head_dim, bool is_bf16);
void thrift_attention_noncausal_mxfp4(
    const void* Q_fp16, const void* K_fp16, const void* V_fp16,
    const void* selected_blocks, int topk_count,
    const void* topk_mask, int topk_word_count,
    const void* Q, const void* K, const void* V,
    const void* S_Q, const void* S_K, const void* S_V,
    void* O, void* rowmax_state, void* rowsum_state,
    int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads, int head_dim, bool is_bf16);
// Implemented in csrc/cuda/sm120/nvfp4/single_query_attention.cu.
void thrift_attention_single_query_nvfp4(
    const void* Q_fp16, const void* K_fp16, const void* V_fp16,
    const int32_t* selected_blocks, int topk_count,
    const void* Q, const void* K, const void* V,
    const void* S_Q, const void* S_K, const void* S_V,
    void* O, void* workspace,
    int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads, int head_dim, bool is_bf16);
// Implemented in csrc/cuda/sm120/mxfp4/single_query_attention.cu.
void thrift_attention_single_query_mxfp4(
    const void* Q_fp16, const void* K_fp16, const void* V_fp16,
    const int32_t* selected_blocks, int topk_count,
    const void* Q, const void* K, const void* V,
    const void* S_Q, const void* S_K, const void* S_V,
    void* O, void* workspace,
    int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads, int head_dim, bool is_bf16);

// Implemented in csrc/cuda/sm120/nvfp4/quantization.cu.
void nvfp4_quantise(void* X, void* X_fp4, void* X_scale, int bs, int seq_len, int head_dim, bool is_bf16);
void nvfp4_quantise_permute_seq(
    void* X, void* X_fp4, void* X_scale, int bs, int seq_len, int head_dim, bool inverse, bool is_bf16);
void nvfp4_quantise_transpose(void* X, void* X_fp4, void* X_scale, int bs, int seq_len, int head_dim, bool is_bf16);
void nvfp4_quantise_transpose_permute_seq(
    void* X, void* X_fp4, void* X_scale, int bs, int seq_len, int head_dim, bool inverse, bool is_bf16);
// Implemented in csrc/cuda/sm120/mxfp4/quantization.cu.
void mxfp4_quantise(void* X, void* X_fp4, void* X_scale, int bs, int seq_len, int head_dim, bool is_bf16);
void mxfp4_quantise_permute_seq(
    void* X, void* X_fp4, void* X_scale, int bs, int seq_len, int head_dim, bool inverse, bool is_bf16);
void mxfp4_quantise_transpose(void* X, void* X_fp4, void* X_scale, int bs, int seq_len, int head_dim, bool is_bf16);
void mxfp4_quantise_transpose_permute_seq(
    void* X, void* X_fp4, void* X_scale, int bs, int seq_len, int head_dim, bool inverse, bool is_bf16);

// Implemented in csrc/cuda/sm120/shared/block_selection.cu.
void block_mean_topk(
    const void* q_mean,
    const void* k_mean,
    void* topk_out,
    int flat_heads,
    int num_q_blocks,
    int num_kv_blocks,
    int head_dim,
    int topk_count,
    bool causal,
    bool is_bf16);
void quest_block_topk(
    const void* q_mean,
    const void* k_min,
    const void* k_max,
    void* topk_out,
    int flat_heads,
    int num_q_blocks,
    int num_kv_blocks,
    int head_dim,
    int topk_count,
    bool causal,
    bool is_bf16);
void single_query_key_mean_topk(
    const void* q_grouped,
    const void* k_mean,
    void* topk_out,
    int flat_heads,
    int groups,
    int num_kv_blocks,
    int k_mean_capacity_blocks,
    int head_dim,
    int topk_count,
    bool is_bf16);
void single_query_quest_topk(
    const void* q_grouped,
    const void* k_min,
    const void* k_max,
    void* topk_out,
    int flat_heads,
    int groups,
    int num_kv_blocks,
    int k_stat_capacity_blocks,
    int head_dim,
    int topk_count,
    bool is_bf16);
// Implemented in csrc/cuda/sm120/shared/block_selection.cu.
cudaError_t local_block_topk(
    int32_t* topk_out,
    int flat_heads,
    int num_q_blocks,
    int num_kv_blocks,
    int topk_count,
    bool causal);
cudaError_t single_query_local_topk(
    int32_t* topk_out,
    int flat_heads,
    int num_kv_blocks,
    int topk_count);
void single_query_key_mean_topk_chunked(
    const void* q_grouped,
    const void* k_mean,
    void* topk_out,
    void* local_scores,
    void* local_indices,
    void* done_counts,
    int flat_heads,
    int groups,
    int num_kv_blocks,
    int k_mean_capacity_blocks,
    int head_dim,
    int topk_count,
    int chunk_count,
    int local_count,
    bool is_bf16);
void pack_topk_mask(
    const int32_t* topk,
    void* topk_mask,
    int row_count,
    int topk_count,
    int total_units,
    int word_count);

static at::Tensor fp4_attention_nvfp4_packed(
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool causal,
    bool is_bf16) {
    check_packed_qkv(q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t);

    const int batch = q_packed.size(0);
    const int num_q_heads = q_packed.size(1);
    const int num_kv_heads = k_packed.size(1);
    const int flat_q_heads = batch * num_q_heads;
    const int q_len = q_packed.size(2);
    const int kv_len = k_packed.size(2);
    const int head_dim = q_packed.size(3) * 2;
    const int kv_capacity = static_cast<int>(k_packed.stride(1)) / (head_dim / 2);
    const at::ScalarType out_dtype = is_bf16 ? at::kBFloat16 : at::kHalf;

    at::Tensor out = at::empty({batch, num_q_heads, q_len, head_dim},
        at::TensorOptions().dtype(out_dtype).device(q_packed.device()));

    if (causal) {
        fp4_attention_causal_nvfp4(
            q_packed.data_ptr(), k_packed.data_ptr(), v_packed_t.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), v_scale_t.data_ptr(),
            out.data_ptr(), flat_q_heads, q_len, kv_len, kv_capacity,
            num_q_heads, num_kv_heads, head_dim, is_bf16);
    } else {
        fp4_attention_noncausal_nvfp4(
            q_packed.data_ptr(), k_packed.data_ptr(), v_packed_t.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), v_scale_t.data_ptr(),
            out.data_ptr(), flat_q_heads, q_len, kv_len, kv_capacity,
            num_q_heads, num_kv_heads, head_dim, is_bf16);
    }

    return out;
}

static at::Tensor fp4_attention_causal_nvfp4_packed(
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool is_bf16 = false) {
    return fp4_attention_nvfp4_packed(
        q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, true, is_bf16);
}

static at::Tensor fp4_attention_noncausal_nvfp4_packed(
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool is_bf16 = false) {
    return fp4_attention_nvfp4_packed(
        q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, false, is_bf16);
}

// Variable-length NVFP4 self-attention (fp4_attention_varlen_kernel). The FP4 inputs hold all sequences in one
// batch row, each starting at a multiple of 64 tokens (cu_seqlens_padded); the output is
// [total_tokens, num_q_heads, head_dim] at the unpadded offsets cu_seqlens. key_bias, if given, is
// [num_q_heads, tokens] floats added to the logits of each key, in the padded token positions.
// The kernel reads, without checking, that cu_seqlens_padded[i] is a multiple of 64 with room for
// round_up(length_i, 64) tokens up to the last entry, at most the token count; that cu_seqlens[-1] is total_tokens;
// that max_seqlen is at least every length; and that V is zero past each sequence before quantization, since its
// scales group 16 tokens (Q and K tokens past a sequence are never read into its output).
// With out_global_scale, the output is instead the NVFP4 input of the next Linear: packed E2M1 values
// [total_tokens, num_q_heads * head_dim / 2] (uint8) and float8_e4m3fn scales in the 128x4 swizzled layout of the
// block-scaled GEMMs, [round_up(total_tokens, 128), round_up(num_q_heads * head_dim / 16, 4)], as vLLM's
// scaled_fp4_quant(out, out_global_scale) writes them; the scale rows past total_tokens are zero.
static std::vector<at::Tensor> fp4_attention_varlen_nvfp4_run(
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    const at::Tensor& cu_seqlens,
    const at::Tensor& cu_seqlens_padded,
    int64_t max_seqlen,
    int64_t total_tokens,
    bool causal,
    bool is_bf16,
    const c10::optional<at::Tensor>& key_bias,
    const c10::optional<at::Tensor>& out_global_scale,
    const c10::optional<at::Tensor>& q_start = c10::nullopt,
    const c10::optional<at::Tensor>& out_start = c10::nullopt,
    bool return_lse = false,
    const c10::optional<at::Tensor>& out_buffer = c10::nullopt,
    const c10::optional<at::Tensor>& out_scales_buffer = c10::nullopt,
    const c10::optional<at::Tensor>& lse_buffer = c10::nullopt,
    const PagedBf16PrefixArgs* paged_prefix = nullptr) {
    check_varlen_inputs(q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, cu_seqlens, cu_seqlens_padded);
    TORCH_CHECK(max_seqlen >= 0 && total_tokens >= 0, "max_seqlen and total_tokens must not be negative");
    const c10::cuda::OptionalCUDAGuard device_guard(q_packed.device());

    const int num_seqs = static_cast<int>(cu_seqlens.numel()) - 1;
    const int num_q_heads = q_packed.size(1);
    const int num_kv_heads = k_packed.size(1);
    const int q_tokens = q_packed.size(2);
    const int k_tokens = k_packed.size(2);
    const int head_dim = q_packed.size(3) * 2;
    const int v_tokens = v_packed_t.size(3) * 2;
    TORCH_CHECK(head_dim == 64 || head_dim == 128, "varlen attention supports head_dim 64 and 128, got ", head_dim);

    const float* bias_ptr = nullptr;
    if (key_bias.has_value()) {
        const at::Tensor& bias = key_bias.value();
        TORCH_CHECK(bias.device() == q_packed.device() && bias.scalar_type() == at::kFloat && bias.is_contiguous(),
                    "key_bias must be a contiguous float32 tensor on the device of q_packed");
        TORCH_CHECK(bias.dim() == 2 && bias.size(0) == num_q_heads && bias.size(1) == k_tokens,
                    "key_bias must be [num_q_heads, tokens]");
        bias_ptr = bias.data_ptr<float>();
    }

    const int* q_start_ptr = nullptr;
    const int* out_start_ptr = nullptr;
    if (q_start.has_value()) {
        const at::Tensor& t = q_start.value();
        TORCH_CHECK(t.device() == q_packed.device() && t.scalar_type() == at::kInt && t.is_contiguous() &&
                    t.numel() >= num_seqs, "q_start must be a contiguous int32 tensor with one entry per sequence");
        q_start_ptr = t.data_ptr<int>();
    }
    if (out_start.has_value()) {
        const at::Tensor& t = out_start.value();
        TORCH_CHECK(t.device() == q_packed.device() && t.scalar_type() == at::kInt && t.is_contiguous() &&
                    t.numel() >= num_seqs, "out_start must be a contiguous int32 tensor with one entry per sequence");
        out_start_ptr = t.data_ptr<int>();
    }

    const auto device = at::TensorOptions().device(q_packed.device());
    std::vector<at::Tensor> outputs;
    void* out_ptr = nullptr;
    uint8_t* fp4_ptr = nullptr;
    uint8_t* sf_ptr = nullptr;
    const float* global_scale_ptr = nullptr;
    if (out_global_scale.has_value()) {
        const at::Tensor& global_scale = out_global_scale.value();
        TORCH_CHECK(global_scale.device() == q_packed.device() && global_scale.scalar_type() == at::kFloat &&
                    global_scale.numel() == 1,
                    "out_global_scale must be a one-element float32 tensor on the device of q_packed");
        const int64_t columns = static_cast<int64_t>(num_q_heads) * head_dim;
        const int64_t sf_rows = (total_tokens + 127) / 128 * 128;
        const int64_t sf_cols = (columns / 16 + 3) / 4 * 4;
        if (out_buffer.has_value()) {  // written in place (a buffer the caller's CUDA graph owns)
            TORCH_CHECK(out_scales_buffer.has_value(), "an NVFP4 output buffer needs a scale buffer");
            const at::Tensor& values = out_buffer.value();
            const at::Tensor& scales = out_scales_buffer.value();
            check_output_buffer(values, q_packed, at::kByte, total_tokens, total_tokens * columns / 2, "out");
            TORCH_CHECK(scales.scalar_type() == at::kFloat8_e4m3fn || scales.scalar_type() == at::kByte,
                        "out_scales must be float8_e4m3fn (or its bytes, uint8)");
            check_output_buffer(scales, q_packed, scales.scalar_type(), sf_rows, sf_rows * sf_cols, "out_scales");
            outputs.push_back(values);
            outputs.push_back(scales);
        } else {
            outputs.push_back(at::empty({total_tokens, columns / 2}, device.dtype(at::kByte)));
            outputs.push_back(at::empty({sf_rows, sf_cols}, device.dtype(at::kFloat8_e4m3fn)));
        }
        fp4_ptr = outputs[0].data_ptr<uint8_t>();
        sf_ptr = static_cast<uint8_t*>(outputs[1].data_ptr());
        global_scale_ptr = global_scale.data_ptr<float>();
        // GEMMs read whole tiles of 128 scale rows, and the kernel writes them only up to total_tokens: zero the last
        // tile first (a tile is contiguous in the swizzled layout)
        if (total_tokens % 128 != 0) {
            outputs[1].view(at::kByte).view(-1).narrow(0, (sf_rows - 128) * sf_cols, 128 * sf_cols).zero_();
        }
    } else {
        TORCH_CHECK(!out_scales_buffer.has_value(), "out_scales is only for the NVFP4 output (out_global_scale)");
        if (out_buffer.has_value()) {
            const at::Tensor& out = out_buffer.value();
            check_output_buffer(out, q_packed, is_bf16 ? at::kBFloat16 : at::kHalf, total_tokens,
                                total_tokens * num_q_heads * head_dim, "out");
            outputs.push_back(out);
        } else {
            outputs.push_back(at::empty({total_tokens, num_q_heads, head_dim},
                                        device.dtype(is_bf16 ? at::kBFloat16 : at::kHalf)));
        }
        out_ptr = outputs[0].data_ptr();
    }
    float* lse_ptr = nullptr;
    if (lse_buffer.has_value()) {
        const at::Tensor& lse = lse_buffer.value();
        check_output_buffer(lse, q_packed, at::kFloat, num_q_heads, num_q_heads * total_tokens, "lse");
        outputs.push_back(lse);
        lse_ptr = lse.data_ptr<float>();
    } else if (return_lse) {
        outputs.push_back(at::empty({num_q_heads, total_tokens}, device.dtype(at::kFloat)));
        lse_ptr = outputs.back().data_ptr<float>();
    }
    if (num_seqs == 0 || total_tokens == 0) {
        return outputs;
    }
    TORCH_CHECK(max_seqlen > 0, "max_seqlen must be positive for a non-empty batch");

    fp4_attention_varlen_nvfp4(
        q_packed.data_ptr(), k_packed.data_ptr(), v_packed_t.data_ptr(),
        q_scale.data_ptr(), k_scale.data_ptr(), v_scale_t.data_ptr(),
        bias_ptr, out_ptr, cu_seqlens.data_ptr<int>(), cu_seqlens_padded.data_ptr<int>(),
        num_seqs, static_cast<int>(max_seqlen), q_tokens, k_tokens, v_tokens,
        num_q_heads, num_kv_heads, head_dim, causal, is_bf16, fp4_ptr, sf_ptr, global_scale_ptr,
        q_start_ptr, out_start_ptr, lse_ptr, static_cast<int>(total_tokens), at::cuda::getCurrentCUDAStream(),
        paged_prefix);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return outputs;
}

static at::Tensor fp4_attention_varlen_nvfp4_packed(
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    const at::Tensor& cu_seqlens,
    const at::Tensor& cu_seqlens_padded,
    int64_t max_seqlen,
    int64_t total_tokens,
    bool causal,
    bool is_bf16,
    const c10::optional<at::Tensor>& key_bias) {
    return fp4_attention_varlen_nvfp4_run(
        q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, cu_seqlens, cu_seqlens_padded,
        max_seqlen, total_tokens, causal, is_bf16, key_bias, c10::nullopt)[0];
}

static std::vector<at::Tensor> fp4_attention_varlen_nvfp4_packed_fp4_out(
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    const at::Tensor& cu_seqlens,
    const at::Tensor& cu_seqlens_padded,
    int64_t max_seqlen,
    int64_t total_tokens,
    bool causal,
    bool is_bf16,
    const c10::optional<at::Tensor>& key_bias,
    const at::Tensor& out_global_scale) {
    return fp4_attention_varlen_nvfp4_run(
        q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, cu_seqlens, cu_seqlens_padded,
        max_seqlen, total_tokens, causal, is_bf16, key_bias, out_global_scale);
}

// The varlen kernel for queries that come after keys a cache holds: the FP4 inputs hold whole sequences (cu_seqlens,
// cu_seqlens_padded: their full lengths), but only the query rows from q_start[seq] on are computed, and row
// q_start[seq] + i is written to output row out_start[seq] + i (out_tokens rows in all). The rows that are written
// are those of fp4_attention_varlen_nvfp4_packed(_fp4_out) on the same inputs. max_seqlen is the most rows a
// sequence computes (its length minus q_start), which sizes the grid; the kernel reads, without checking, that
// 0 <= q_start[seq] <= length and that the output rows fit in out_tokens.
// Returns the output (or the NVFP4 values and scales, with out_global_scale), then, with return_lse or lse, the
// log-sum-exp [num_q_heads, out_tokens] of the logits the kernel computes, key_bias included: when Q and K were
// smoothed (q - c_q, k - c_k, key_bias c_q . (k - c_k)), the row constant scale * q . c_k is not part of it.
// out (and out_scales for the NVFP4 output) and lse, if given, are written in place instead of allocated, so that a
// captured CUDA graph can reuse them.
// With prefix_len (causal bfloat16 only), keys 0 .. prefix_len[seq] - 1 of each sequence come in bfloat16 from a
// paged cache instead (key_cache and value_cache [blocks, block_size, num_kv_heads, head_dim], block_table
// [num_seqs, max_blocks]) and are attended after the sequence's FP4 keys, which then start at its key
// prefix_len[seq], in the same online softmax; q_bf16 holds the rows' queries. key_bias applies to the FP4 keys
// only; key_mean, the vector subtracted from the keys before FP4 quantization, brings the prefix logits to the FP4
// logits' frame. The kernel reads, without checking, that prefix_len[seq] <= block_table.size(1) * block_size and
// that the blocks exist.
static std::vector<at::Tensor> fp4_attention_varlen_nvfp4_cached(
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    const at::Tensor& cu_seqlens,
    const at::Tensor& cu_seqlens_padded,
    int64_t max_seqlen,
    int64_t out_tokens,
    bool causal,
    bool is_bf16,
    const c10::optional<at::Tensor>& key_bias,
    const at::Tensor& q_start,
    const at::Tensor& out_start,
    const c10::optional<at::Tensor>& out_global_scale,
    bool return_lse,
    const c10::optional<at::Tensor>& out,
    const c10::optional<at::Tensor>& out_scales,
    const c10::optional<at::Tensor>& lse,
    const c10::optional<at::Tensor>& q_bf16,
    const c10::optional<at::Tensor>& key_cache,
    const c10::optional<at::Tensor>& value_cache,
    const c10::optional<at::Tensor>& block_table,
    const c10::optional<at::Tensor>& prefix_len,
    const c10::optional<at::Tensor>& key_mean) {
    if (!prefix_len.has_value()) {
        TORCH_CHECK(!q_bf16.has_value() && !key_cache.has_value() && !value_cache.has_value() &&
                    !block_table.has_value() && !key_mean.has_value(),
                    "q_bf16, key_cache, value_cache, block_table and key_mean are only used with prefix_len");
        return fp4_attention_varlen_nvfp4_run(
            q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, cu_seqlens, cu_seqlens_padded,
            max_seqlen, out_tokens, causal, is_bf16, key_bias, out_global_scale, q_start, out_start, return_lse, out,
            out_scales, lse);
    }
    // the bfloat16 prefix: keys 0 .. prefix_len[seq] - 1 of each sequence from the paged cache, after its FP4 keys
    TORCH_CHECK(is_bf16 && causal, "a bfloat16 prefix is for causal bfloat16 attention");
    TORCH_CHECK(q_bf16.has_value() && key_cache.has_value() && value_cache.has_value() && block_table.has_value(),
                "a bfloat16 prefix needs q_bf16, key_cache, value_cache and block_table");
    const int num_seqs = static_cast<int>(cu_seqlens.numel()) - 1;
    const int num_q_heads = q_packed.size(1);
    const int num_kv_heads = k_packed.size(1);
    const int head_dim = q_packed.size(3) * 2;
    const at::Tensor& q = q_bf16.value();
    const at::Tensor& kc = key_cache.value();
    const at::Tensor& vc = value_cache.value();
    const at::Tensor& table = block_table.value();
    const at::Tensor& lengths = prefix_len.value();
    for (const at::Tensor* t : {&q, &kc, &vc, &table, &lengths}) {
        TORCH_CHECK(t->device() == q_packed.device(), "the prefix inputs must be on the device of q_packed");
    }
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && q.is_contiguous() && q.dim() == 3 && q.size(0) >= out_tokens &&
                q.size(1) == num_q_heads && q.size(2) == head_dim &&
                reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0,
                "q_bf16 must be [out_tokens, num_q_heads, head_dim] contiguous, 16-byte aligned bfloat16");
    for (const at::Tensor* cache : {&kc, &vc}) {
        TORCH_CHECK(cache->scalar_type() == at::kBFloat16 && cache->dim() == 4 && cache->size(1) > 0 &&
                    cache->size(2) == num_kv_heads && cache->size(3) == head_dim && cache->stride(3) == 1 &&
                    cache->stride(2) % 8 == 0 && cache->stride(1) % 8 == 0 && cache->stride(0) % 8 == 0 &&
                    reinterpret_cast<uintptr_t>(cache->data_ptr()) % 16 == 0,
                    "key_cache and value_cache must be [blocks, block_size, num_kv_heads, head_dim] bfloat16 views "
                    "with 16-byte aligned rows");
    }
    TORCH_CHECK(kc.sizes() == vc.sizes() && kc.strides() == vc.strides(),
                "key_cache and value_cache must have the same shape and strides");
    TORCH_CHECK(table.scalar_type() == at::kInt && table.dim() == 2 && table.stride(1) == 1 &&
                table.size(0) >= num_seqs, "block_table must be [num_seqs, max_blocks] int32 with unit column stride");
    TORCH_CHECK(lengths.scalar_type() == at::kInt && lengths.is_contiguous() && lengths.numel() >= num_seqs,
                "prefix_len must be int32 with one entry per sequence");
    PagedBf16PrefixArgs args;
    if (key_mean.has_value()) {
        const at::Tensor& mean = key_mean.value();
        TORCH_CHECK(mean.device() == q_packed.device() && mean.scalar_type() == at::kFloat && mean.is_contiguous() &&
                    mean.dim() == 2 && mean.size(0) == num_kv_heads && mean.size(1) == head_dim,
                    "key_mean must be [num_kv_heads, head_dim] float32");
        args.key_mean = mean.data_ptr<float>();
    }
    args.q_bf16 = q.data_ptr();
    args.key_cache = kc.data_ptr();
    args.value_cache = vc.data_ptr();
    args.block_stride = kc.stride(0);
    args.token_stride = kc.stride(1);
    args.head_stride = kc.stride(2);
    args.block_table = table.data_ptr<int>();
    args.block_table_stride = static_cast<int>(table.stride(0));
    args.block_size = static_cast<int>(kc.size(1));
    args.prefix_len = lengths.data_ptr<int>();
    return fp4_attention_varlen_nvfp4_run(
        q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, cu_seqlens, cu_seqlens_padded,
        max_seqlen, out_tokens, causal, is_bf16, key_bias, out_global_scale, q_start, out_start, return_lse, out,
        out_scales, lse, &args);
}

static at::Tensor fp4_attention_mxfp4_packed(
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool causal,
    bool is_bf16) {
    check_packed_qkv(q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t);

    const int batch = q_packed.size(0);
    const int num_q_heads = q_packed.size(1);
    const int num_kv_heads = k_packed.size(1);
    const int flat_q_heads = batch * num_q_heads;
    const int q_len = q_packed.size(2);
    const int kv_len = k_packed.size(2);
    const int head_dim = q_packed.size(3) * 2;
    const int kv_capacity = static_cast<int>(k_packed.stride(1)) / (head_dim / 2);
    const at::ScalarType out_dtype = is_bf16 ? at::kBFloat16 : at::kHalf;

    at::Tensor out = at::empty({batch, num_q_heads, q_len, head_dim},
        at::TensorOptions().dtype(out_dtype).device(q_packed.device()));

    if (causal) {
        fp4_attention_causal_mxfp4(
            q_packed.data_ptr(), k_packed.data_ptr(), v_packed_t.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), v_scale_t.data_ptr(),
            out.data_ptr(), flat_q_heads, q_len, kv_len, kv_capacity,
            num_q_heads, num_kv_heads, head_dim, is_bf16);
    } else {
        fp4_attention_noncausal_mxfp4(
            q_packed.data_ptr(), k_packed.data_ptr(), v_packed_t.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), v_scale_t.data_ptr(),
            out.data_ptr(), flat_q_heads, q_len, kv_len, kv_capacity,
            num_q_heads, num_kv_heads, head_dim, is_bf16);
    }

    return out;
}

static at::Tensor fp4_attention_causal_mxfp4_packed(
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool is_bf16 = false) {
    return fp4_attention_mxfp4_packed(
        q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, true, is_bf16);
}

static at::Tensor fp4_attention_noncausal_mxfp4_packed(
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool is_bf16 = false) {
    return fp4_attention_mxfp4_packed(
        q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, false, is_bf16);
}

static at::Tensor fp4_attention_single_query_nvfp4_packed(
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool is_bf16 = false) {
    check_packed_qkv(q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t);

    const int batch = q_packed.size(0);
    const int num_q_heads = q_packed.size(1);
    const int num_kv_heads = k_packed.size(1);
    const int flat_q_heads = batch * num_q_heads;
    const int q_len = q_packed.size(2);
    const int kv_len = k_packed.size(2);
    const int head_dim = q_packed.size(3) * 2;
    const int kv_capacity = static_cast<int>(k_packed.stride(1)) / (head_dim / 2);
    const at::ScalarType out_dtype = is_bf16 ? at::kBFloat16 : at::kHalf;
    TORCH_CHECK(q_len >= 1 && q_len <= 16,
                "single-query FP4 attention expects grouped query length in [1, 16], got ",
                q_len);

    at::Tensor out = at::empty({batch, num_q_heads, q_len, head_dim},
        at::TensorOptions().dtype(out_dtype).device(q_packed.device()));

    constexpr int block_kv = 64;
    const int total_kv_blocks = (kv_len + block_kv - 1) / block_kv;
    const bool use_split = total_kv_blocks >= 128;

    auto dispatch = [&](void* workspace) {
        fp4_attention_single_query_nvfp4(
            q_packed.data_ptr(), k_packed.data_ptr(), v_packed_t.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), v_scale_t.data_ptr(),
            out.data_ptr(), workspace,
            flat_q_heads, q_len, kv_len, kv_capacity,
            num_q_heads, num_kv_heads, head_dim, is_bf16);
    };

    if (!use_split) {
        dispatch(nullptr);
        return out;
    }

    int num_kv_splits = std::max(1, std::min(total_kv_blocks, (256 + flat_q_heads - 1) / flat_q_heads));
    num_kv_splits = std::min(num_kv_splits, total_kv_blocks);
    const int64_t workspace_elems =
        static_cast<int64_t>(flat_q_heads) * num_kv_splits * 16 * (head_dim + 2) + flat_q_heads;
    at::Tensor workspace = at::empty({workspace_elems},
        at::TensorOptions().dtype(at::kFloat).device(q_packed.device()));
    dispatch(workspace.data_ptr());
    return out;
}

static at::Tensor fp4_attention_single_query_mxfp4_packed(
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool is_bf16 = false) {
    check_packed_qkv(q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t);

    const int batch = q_packed.size(0);
    const int num_q_heads = q_packed.size(1);
    const int num_kv_heads = k_packed.size(1);
    const int flat_q_heads = batch * num_q_heads;
    const int q_len = q_packed.size(2);
    const int kv_len = k_packed.size(2);
    const int head_dim = q_packed.size(3) * 2;
    const int kv_capacity = static_cast<int>(k_packed.stride(1)) / (head_dim / 2);
    const at::ScalarType out_dtype = is_bf16 ? at::kBFloat16 : at::kHalf;
    TORCH_CHECK(q_len >= 1 && q_len <= 16,
                "single-query FP4 attention expects grouped query length in [1, 16], got ",
                q_len);

    at::Tensor out = at::empty({batch, num_q_heads, q_len, head_dim},
        at::TensorOptions().dtype(out_dtype).device(q_packed.device()));

    constexpr int block_kv = 64;
    const int total_kv_blocks = (kv_len + block_kv - 1) / block_kv;
    const bool use_split = total_kv_blocks >= 128;

    auto dispatch = [&](void* workspace) {
        fp4_attention_single_query_mxfp4(
            q_packed.data_ptr(), k_packed.data_ptr(), v_packed_t.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), v_scale_t.data_ptr(),
            out.data_ptr(), workspace,
            flat_q_heads, q_len, kv_len, kv_capacity,
            num_q_heads, num_kv_heads, head_dim, is_bf16);
    };

    if (!use_split) {
        dispatch(nullptr);
        return out;
    }

    int num_kv_splits = std::max(1, std::min(total_kv_blocks, (256 + flat_q_heads - 1) / flat_q_heads));
    num_kv_splits = std::min(num_kv_splits, total_kv_blocks);
    const int64_t workspace_elems =
        static_cast<int64_t>(flat_q_heads) * num_kv_splits * 16 * (head_dim + 2) + flat_q_heads;
    at::Tensor workspace = at::empty({workspace_elems},
        at::TensorOptions().dtype(at::kFloat).device(q_packed.device()));
    dispatch(workspace.data_ptr());
    return out;
}

static at::Tensor thrift_attention_nvfp4_packed(
    const at::Tensor& q_hi,
    const at::Tensor& k_hi,
    const at::Tensor& v_hi,
    const at::Tensor& selected_blocks,
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool causal,
    bool is_bf16) {
    check_hi_qkv(q_hi, k_hi, v_hi);
    check_packed_qkv(q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t);
    TORCH_CHECK(selected_blocks.is_cuda(), "selected_blocks must be a CUDA tensor");
    TORCH_CHECK(selected_blocks.dtype() == at::kInt, "selected_blocks must be int32");
    TORCH_CHECK(selected_blocks.is_contiguous(), "selected_blocks must be contiguous");
    TORCH_CHECK(selected_blocks.dim() == 3,
                "selected_blocks must be [batch * heads, q_blocks, top_k]");

    const int batch = q_packed.size(0);
    const int num_q_heads = q_packed.size(1);
    const int num_kv_heads = k_packed.size(1);
    const int flat_q_heads = batch * num_q_heads;
    const int q_len = q_packed.size(2);
    const int kv_len = k_packed.size(2);
    const int head_dim = q_packed.size(3) * 2;
    const int block_q = (head_dim == 256) ? 128 : 64;
    const int num_q_blocks = (q_len + block_q - 1) / block_q;
    const int topk_count = selected_blocks.size(2);
    const int kv_capacity = static_cast<int>(k_packed.stride(1)) / (head_dim / 2);
    const at::ScalarType out_dtype = is_bf16 ? at::kBFloat16 : at::kHalf;
    TORCH_CHECK(selected_blocks.size(0) == flat_q_heads,
                "selected_blocks first dimension must equal batch * Q heads");
    TORCH_CHECK(selected_blocks.size(1) == num_q_blocks,
                "selected_blocks second dimension must equal number of Q blocks");

    if (topk_count == 0) {
        return fp4_attention_nvfp4_packed(
            q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, causal, is_bf16);
    }

    constexpr int topk_unit_tokens = 64;
    const int total_topk_units = (kv_len + topk_unit_tokens - 1) / topk_unit_tokens;
    const int topk_word_count = (total_topk_units + 63) / 64;
    TORCH_CHECK(topk_word_count <= 32,
                "ThriftAttention supports at most 2048 64-token KV blocks, got ",
                total_topk_units);

    auto mask_opts = at::TensorOptions().dtype(at::kLong).device(selected_blocks.device());
    at::Tensor topk_mask = at::empty({flat_q_heads, num_q_blocks, topk_word_count}, mask_opts);
    pack_topk_mask(
        static_cast<const int32_t*>(selected_blocks.data_ptr()),
        topk_mask.data_ptr(),
        flat_q_heads * num_q_blocks, topk_count, total_topk_units, topk_word_count);

    at::Tensor out = at::empty({batch, num_q_heads, q_len, head_dim},
        at::TensorOptions().dtype(out_dtype).device(q_packed.device()));
    auto state_opts = at::TensorOptions().dtype(at::kFloat).device(q_packed.device());
    at::Tensor rowmax_state = at::empty({batch, num_q_heads, q_len}, state_opts);
    at::Tensor rowsum_state = at::empty({batch, num_q_heads, q_len}, state_opts);

    if (causal) {
        thrift_attention_causal_nvfp4(
            q_hi.data_ptr(), k_hi.data_ptr(), v_hi.data_ptr(),
            selected_blocks.data_ptr(), topk_count,
            topk_mask.data_ptr(), topk_word_count,
            q_packed.data_ptr(), k_packed.data_ptr(), v_packed_t.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), v_scale_t.data_ptr(),
            out.data_ptr(), rowmax_state.data_ptr(), rowsum_state.data_ptr(),
            flat_q_heads, q_len, kv_len, kv_capacity,
            num_q_heads, num_kv_heads, head_dim, is_bf16);
    } else {
        thrift_attention_noncausal_nvfp4(
            q_hi.data_ptr(), k_hi.data_ptr(), v_hi.data_ptr(),
            selected_blocks.data_ptr(), topk_count,
            topk_mask.data_ptr(), topk_word_count,
            q_packed.data_ptr(), k_packed.data_ptr(), v_packed_t.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), v_scale_t.data_ptr(),
            out.data_ptr(), rowmax_state.data_ptr(), rowsum_state.data_ptr(),
            flat_q_heads, q_len, kv_len, kv_capacity,
            num_q_heads, num_kv_heads, head_dim, is_bf16);
    }

    return out;
}

static at::Tensor thrift_attention_causal_nvfp4_packed(
    const at::Tensor& q_hi,
    const at::Tensor& k_hi,
    const at::Tensor& v_hi,
    const at::Tensor& selected_blocks,
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool is_bf16 = false) {
    return thrift_attention_nvfp4_packed(
        q_hi, k_hi, v_hi, selected_blocks,
        q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, true, is_bf16);
}

static at::Tensor thrift_attention_noncausal_nvfp4_packed(
    const at::Tensor& q_hi,
    const at::Tensor& k_hi,
    const at::Tensor& v_hi,
    const at::Tensor& selected_blocks,
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool is_bf16 = false) {
    return thrift_attention_nvfp4_packed(
        q_hi, k_hi, v_hi, selected_blocks,
        q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, false, is_bf16);
}

static at::Tensor thrift_attention_mxfp4_packed(
    const at::Tensor& q_hi,
    const at::Tensor& k_hi,
    const at::Tensor& v_hi,
    const at::Tensor& selected_blocks,
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool causal,
    bool is_bf16) {
    check_hi_qkv(q_hi, k_hi, v_hi);
    check_packed_qkv(q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t);
    TORCH_CHECK(selected_blocks.is_cuda(), "selected_blocks must be a CUDA tensor");
    TORCH_CHECK(selected_blocks.dtype() == at::kInt, "selected_blocks must be int32");
    TORCH_CHECK(selected_blocks.is_contiguous(), "selected_blocks must be contiguous");
    TORCH_CHECK(selected_blocks.dim() == 3,
                "selected_blocks must be [batch * heads, q_blocks, top_k]");

    const int batch = q_packed.size(0);
    const int num_q_heads = q_packed.size(1);
    const int num_kv_heads = k_packed.size(1);
    const int flat_q_heads = batch * num_q_heads;
    const int q_len = q_packed.size(2);
    const int kv_len = k_packed.size(2);
    const int head_dim = q_packed.size(3) * 2;
    const int block_q = (head_dim == 256) ? 128 : 64;
    const int num_q_blocks = (q_len + block_q - 1) / block_q;
    const int topk_count = selected_blocks.size(2);
    const int kv_capacity = static_cast<int>(k_packed.stride(1)) / (head_dim / 2);
    const at::ScalarType out_dtype = is_bf16 ? at::kBFloat16 : at::kHalf;
    TORCH_CHECK(selected_blocks.size(0) == flat_q_heads,
                "selected_blocks first dimension must equal batch * Q heads");
    TORCH_CHECK(selected_blocks.size(1) == num_q_blocks,
                "selected_blocks second dimension must equal number of Q blocks");

    if (topk_count == 0) {
        return fp4_attention_mxfp4_packed(
            q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, causal, is_bf16);
    }

    constexpr int topk_unit_tokens = 64;
    const int total_topk_units = (kv_len + topk_unit_tokens - 1) / topk_unit_tokens;
    const int topk_word_count = (total_topk_units + 63) / 64;
    TORCH_CHECK(topk_word_count <= 32,
                "ThriftAttention supports at most 2048 64-token KV blocks, got ",
                total_topk_units);

    auto mask_opts = at::TensorOptions().dtype(at::kLong).device(selected_blocks.device());
    at::Tensor topk_mask = at::empty({flat_q_heads, num_q_blocks, topk_word_count}, mask_opts);
    pack_topk_mask(
        static_cast<const int32_t*>(selected_blocks.data_ptr()),
        topk_mask.data_ptr(),
        flat_q_heads * num_q_blocks, topk_count, total_topk_units, topk_word_count);

    at::Tensor out = at::empty({batch, num_q_heads, q_len, head_dim},
        at::TensorOptions().dtype(out_dtype).device(q_packed.device()));
    auto state_opts = at::TensorOptions().dtype(at::kFloat).device(q_packed.device());
    at::Tensor rowmax_state = at::empty({batch, num_q_heads, q_len}, state_opts);
    at::Tensor rowsum_state = at::empty({batch, num_q_heads, q_len}, state_opts);

    if (causal) {
        thrift_attention_causal_mxfp4(
            q_hi.data_ptr(), k_hi.data_ptr(), v_hi.data_ptr(),
            selected_blocks.data_ptr(), topk_count,
            topk_mask.data_ptr(), topk_word_count,
            q_packed.data_ptr(), k_packed.data_ptr(), v_packed_t.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), v_scale_t.data_ptr(),
            out.data_ptr(), rowmax_state.data_ptr(), rowsum_state.data_ptr(),
            flat_q_heads, q_len, kv_len, kv_capacity,
            num_q_heads, num_kv_heads, head_dim, is_bf16);
    } else {
        thrift_attention_noncausal_mxfp4(
            q_hi.data_ptr(), k_hi.data_ptr(), v_hi.data_ptr(),
            selected_blocks.data_ptr(), topk_count,
            topk_mask.data_ptr(), topk_word_count,
            q_packed.data_ptr(), k_packed.data_ptr(), v_packed_t.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), v_scale_t.data_ptr(),
            out.data_ptr(), rowmax_state.data_ptr(), rowsum_state.data_ptr(),
            flat_q_heads, q_len, kv_len, kv_capacity,
            num_q_heads, num_kv_heads, head_dim, is_bf16);
    }

    return out;
}

static at::Tensor thrift_attention_causal_mxfp4_packed(
    const at::Tensor& q_hi,
    const at::Tensor& k_hi,
    const at::Tensor& v_hi,
    const at::Tensor& selected_blocks,
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool is_bf16 = false) {
    return thrift_attention_mxfp4_packed(
        q_hi, k_hi, v_hi, selected_blocks,
        q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, true, is_bf16);
}

static at::Tensor thrift_attention_noncausal_mxfp4_packed(
    const at::Tensor& q_hi,
    const at::Tensor& k_hi,
    const at::Tensor& v_hi,
    const at::Tensor& selected_blocks,
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool is_bf16 = false) {
    return thrift_attention_mxfp4_packed(
        q_hi, k_hi, v_hi, selected_blocks,
        q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, false, is_bf16);
}

static at::Tensor thrift_attention_single_query_nvfp4_packed(
    const at::Tensor& q_hi,
    const at::Tensor& k_hi,
    const at::Tensor& v_hi,
    const at::Tensor& selected_blocks,
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool is_bf16 = false) {
    check_hi_qkv(q_hi, k_hi, v_hi);
    check_packed_qkv(q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t);
    TORCH_CHECK(selected_blocks.is_cuda(), "selected_blocks must be a CUDA tensor");
    TORCH_CHECK(selected_blocks.dtype() == at::kInt, "selected_blocks must be int32");
    TORCH_CHECK(selected_blocks.is_contiguous(), "selected_blocks must be contiguous");
    TORCH_CHECK(selected_blocks.dim() == 2,
                "selected_blocks must be [batch * grouped_heads, top_k]");

    const int batch = q_packed.size(0);
    const int num_q_heads = q_packed.size(1);
    const int num_kv_heads = k_packed.size(1);
    const int flat_q_heads = batch * num_q_heads;
    const int q_len = q_packed.size(2);
    const int kv_len = k_packed.size(2);
    const int head_dim = q_packed.size(3) * 2;
    const int topk_count = selected_blocks.size(1);
    const int kv_capacity = static_cast<int>(k_packed.stride(1)) / (head_dim / 2);
    const at::ScalarType out_dtype = is_bf16 ? at::kBFloat16 : at::kHalf;
    TORCH_CHECK(q_len >= 1 && q_len <= 16,
                "single-query ThriftAttention expects grouped query length in [1, 16], got ",
                q_len);
    TORCH_CHECK(q_hi.size(0) == batch && q_hi.size(1) == num_q_heads &&
                q_hi.size(2) == q_len && q_hi.size(3) == head_dim,
                "q_hi shape must match packed Q");
    TORCH_CHECK(selected_blocks.size(0) == flat_q_heads,
                "selected_blocks first dimension must equal batch * grouped_heads");

    constexpr int block_kv = 64;
    const int total_kv_blocks = (kv_len + block_kv - 1) / block_kv;
    TORCH_CHECK(topk_count >= 0 && topk_count <= total_kv_blocks,
                "selected block count must be in [0, number of KV blocks]");

    if (topk_count == 0) {
        return fp4_attention_single_query_nvfp4_packed(
            q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, is_bf16);
    }

    at::Tensor out = at::empty({batch, num_q_heads, q_len, head_dim},
        at::TensorOptions().dtype(out_dtype).device(q_packed.device()));
    const bool use_split = total_kv_blocks >= 64;

    auto dispatch = [&](void* workspace) {
        thrift_attention_single_query_nvfp4(
            q_hi.data_ptr(), k_hi.data_ptr(), v_hi.data_ptr(),
            static_cast<const int32_t*>(selected_blocks.data_ptr()),
            topk_count,
            q_packed.data_ptr(), k_packed.data_ptr(), v_packed_t.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), v_scale_t.data_ptr(),
            out.data_ptr(), workspace,
            flat_q_heads, q_len, kv_len, kv_capacity,
            num_q_heads, num_kv_heads, head_dim, is_bf16);
    };

    if (!use_split) {
        dispatch(nullptr);
        return out;
    }

    const int target_split_ctas = single_query_mixed_target_split_ctas(total_kv_blocks);
    int num_kv_splits =
        std::max(1, std::min(total_kv_blocks, (target_split_ctas + flat_q_heads - 1) / flat_q_heads));
    num_kv_splits = std::min(num_kv_splits, total_kv_blocks);
    const int64_t workspace_elems =
        static_cast<int64_t>(flat_q_heads) * num_kv_splits * q_len * (head_dim + 2) + flat_q_heads;
    at::Tensor workspace = at::empty({workspace_elems},
        at::TensorOptions().dtype(at::kFloat).device(q_packed.device()));
    dispatch(workspace.data_ptr());
    return out;
}

static at::Tensor thrift_attention_single_query_mxfp4_packed(
    const at::Tensor& q_hi,
    const at::Tensor& k_hi,
    const at::Tensor& v_hi,
    const at::Tensor& selected_blocks,
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool is_bf16 = false) {
    check_hi_qkv(q_hi, k_hi, v_hi);
    check_packed_qkv(q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t);
    TORCH_CHECK(selected_blocks.is_cuda(), "selected_blocks must be a CUDA tensor");
    TORCH_CHECK(selected_blocks.dtype() == at::kInt, "selected_blocks must be int32");
    TORCH_CHECK(selected_blocks.is_contiguous(), "selected_blocks must be contiguous");
    TORCH_CHECK(selected_blocks.dim() == 2,
                "selected_blocks must be [batch * grouped_heads, top_k]");

    const int batch = q_packed.size(0);
    const int num_q_heads = q_packed.size(1);
    const int num_kv_heads = k_packed.size(1);
    const int flat_q_heads = batch * num_q_heads;
    const int q_len = q_packed.size(2);
    const int kv_len = k_packed.size(2);
    const int head_dim = q_packed.size(3) * 2;
    const int topk_count = selected_blocks.size(1);
    const int kv_capacity = static_cast<int>(k_packed.stride(1)) / (head_dim / 2);
    const at::ScalarType out_dtype = is_bf16 ? at::kBFloat16 : at::kHalf;
    TORCH_CHECK(q_len >= 1 && q_len <= 16,
                "single-query ThriftAttention expects grouped query length in [1, 16], got ",
                q_len);
    TORCH_CHECK(q_hi.size(0) == batch && q_hi.size(1) == num_q_heads &&
                q_hi.size(2) == q_len && q_hi.size(3) == head_dim,
                "q_hi shape must match packed Q");
    TORCH_CHECK(selected_blocks.size(0) == flat_q_heads,
                "selected_blocks first dimension must equal batch * grouped_heads");

    constexpr int block_kv = 64;
    const int total_kv_blocks = (kv_len + block_kv - 1) / block_kv;
    TORCH_CHECK(topk_count >= 0 && topk_count <= total_kv_blocks,
                "selected block count must be in [0, number of KV blocks]");

    if (topk_count == 0) {
        return fp4_attention_single_query_mxfp4_packed(
            q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, is_bf16);
    }

    at::Tensor out = at::empty({batch, num_q_heads, q_len, head_dim},
        at::TensorOptions().dtype(out_dtype).device(q_packed.device()));
    const bool use_split = total_kv_blocks >= 64;

    auto dispatch = [&](void* workspace) {
        thrift_attention_single_query_mxfp4(
            q_hi.data_ptr(), k_hi.data_ptr(), v_hi.data_ptr(),
            static_cast<const int32_t*>(selected_blocks.data_ptr()), topk_count,
            q_packed.data_ptr(), k_packed.data_ptr(), v_packed_t.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), v_scale_t.data_ptr(),
            out.data_ptr(), workspace,
            flat_q_heads, q_len, kv_len, kv_capacity,
            num_q_heads, num_kv_heads, head_dim, is_bf16);
    };

    if (!use_split) {
        dispatch(nullptr);
        return out;
    }

    const int target_split_ctas = single_query_mixed_target_split_ctas(total_kv_blocks);
    int num_kv_splits =
        std::max(1, std::min(total_kv_blocks, (target_split_ctas + flat_q_heads - 1) / flat_q_heads));
    num_kv_splits = std::min(num_kv_splits, total_kv_blocks);
    const int64_t workspace_elems =
        static_cast<int64_t>(flat_q_heads) * num_kv_splits * q_len * (head_dim + 2) + flat_q_heads;
    at::Tensor workspace = at::empty({workspace_elems},
        at::TensorOptions().dtype(at::kFloat).device(q_packed.device()));
    dispatch(workspace.data_ptr());
    return out;
}

static std::vector<at::Tensor> nvfp4_quantize(const at::Tensor& x, bool is_bf16 = false) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D [batch, heads, seq, head_dim]");

    const int batch = x.size(0);
    const int heads = x.size(1);
    const int seq_len = x.size(2);
    const int head_dim = x.size(3);
    TORCH_CHECK(head_dim == 64 || head_dim == 128 || head_dim == 256,
                "head_dim must be 64, 128, or 256, got ", head_dim);

    auto opts_u8 = at::TensorOptions().dtype(at::kByte).device(x.device());
    auto opts_f8 = at::TensorOptions().dtype(at::kFloat8_e4m3fn).device(x.device());
    at::Tensor x_packed = at::empty({batch, heads, seq_len, head_dim / 2}, opts_u8);
    at::Tensor x_scale = at::empty({batch, heads, seq_len, head_dim / 16}, opts_f8);

    nvfp4_quantise(
        x.data_ptr(), x_packed.data_ptr(), x_scale.data_ptr(),
        batch * heads, seq_len, head_dim, is_bf16);

    return {x_packed, x_scale};
}

static std::vector<at::Tensor> nvfp4_quantize_permuted(const at::Tensor& x, bool is_bf16 = false) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D [batch, heads, seq, head_dim]");

    const int batch = x.size(0);
    const int heads = x.size(1);
    const int seq_len = x.size(2);
    const int head_dim = x.size(3);
    TORCH_CHECK(head_dim == 64 || head_dim == 128 || head_dim == 256,
                "head_dim must be 64, 128, or 256, got ", head_dim);

    auto opts_u8 = at::TensorOptions().dtype(at::kByte).device(x.device());
    auto opts_f8 = at::TensorOptions().dtype(at::kFloat8_e4m3fn).device(x.device());
    at::Tensor x_packed = at::empty({batch, heads, seq_len, head_dim / 2}, opts_u8);
    at::Tensor x_scale = at::empty({batch, heads, seq_len, head_dim / 16}, opts_f8);

    nvfp4_quantise_permute_seq(
        x.data_ptr(), x_packed.data_ptr(), x_scale.data_ptr(),
        batch * heads, seq_len, head_dim, false, is_bf16);

    return {x_packed, x_scale};
}

static std::vector<at::Tensor> nvfp4_quantize_transposed(const at::Tensor& x, bool is_bf16 = false) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D [batch, heads, seq, head_dim]");

    const int batch = x.size(0);
    const int heads = x.size(1);
    const int seq_len = x.size(2);
    const int head_dim = x.size(3);
    TORCH_CHECK(head_dim == 64 || head_dim == 128 || head_dim == 256,
                "head_dim must be 64, 128, or 256, got ", head_dim);

    constexpr int seq_block = 128;
    const int padded_seq = ((seq_len + seq_block - 1) / seq_block) * seq_block;
    auto opts_u8 = at::TensorOptions().dtype(at::kByte).device(x.device());
    auto opts_f8 = at::TensorOptions().dtype(at::kFloat8_e4m3fn).device(x.device());
    at::Tensor x_packed = at::empty({batch, heads, head_dim, padded_seq / 2}, opts_u8);
    at::Tensor x_scale = at::empty({batch, heads, head_dim, padded_seq / 16}, opts_f8);

    nvfp4_quantise_transpose(
        x.data_ptr(), x_packed.data_ptr(), x_scale.data_ptr(),
        batch * heads, seq_len, head_dim, is_bf16);

    return {x_packed, x_scale};
}

static std::vector<at::Tensor> nvfp4_quantize_transposed_permuted(const at::Tensor& x, bool is_bf16 = false) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D [batch, heads, seq, head_dim]");

    const int batch = x.size(0);
    const int heads = x.size(1);
    const int seq_len = x.size(2);
    const int head_dim = x.size(3);
    TORCH_CHECK(head_dim == 64 || head_dim == 128 || head_dim == 256,
                "head_dim must be 64, 128, or 256, got ", head_dim);

    constexpr int seq_block = 128;
    const int padded_seq = ((seq_len + seq_block - 1) / seq_block) * seq_block;
    auto opts_u8 = at::TensorOptions().dtype(at::kByte).device(x.device());
    auto opts_f8 = at::TensorOptions().dtype(at::kFloat8_e4m3fn).device(x.device());
    at::Tensor x_packed = at::empty({batch, heads, head_dim, padded_seq / 2}, opts_u8);
    at::Tensor x_scale = at::empty({batch, heads, head_dim, padded_seq / 16}, opts_f8);

    nvfp4_quantise_transpose_permute_seq(
        x.data_ptr(), x_packed.data_ptr(), x_scale.data_ptr(),
        batch * heads, seq_len, head_dim, false, is_bf16);

    return {x_packed, x_scale};
}

static std::vector<at::Tensor> mxfp4_quantize(const at::Tensor& x, bool is_bf16 = false) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D [batch, heads, seq, head_dim]");

    const int batch = x.size(0);
    const int heads = x.size(1);
    const int seq_len = x.size(2);
    const int head_dim = x.size(3);
    TORCH_CHECK(head_dim == 64 || head_dim == 128 || head_dim == 256,
                "head_dim must be 64, 128, or 256, got ", head_dim);

    auto opts_u8 = at::TensorOptions().dtype(at::kByte).device(x.device());
    auto opts_e8m0 = at::TensorOptions().dtype(at::kFloat8_e8m0fnu).device(x.device());
    at::Tensor x_packed = at::empty({batch, heads, seq_len, head_dim / 2}, opts_u8);
    at::Tensor x_scale = at::empty({batch, heads, seq_len, head_dim / 32}, opts_e8m0);

    mxfp4_quantise(
        x.data_ptr(), x_packed.data_ptr(), x_scale.data_ptr(),
        batch * heads, seq_len, head_dim, is_bf16);

    return {x_packed, x_scale};
}

static std::vector<at::Tensor> mxfp4_quantize_permuted(const at::Tensor& x, bool is_bf16 = false) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D [batch, heads, seq, head_dim]");

    const int batch = x.size(0);
    const int heads = x.size(1);
    const int seq_len = x.size(2);
    const int head_dim = x.size(3);
    TORCH_CHECK(head_dim == 64 || head_dim == 128 || head_dim == 256,
                "head_dim must be 64, 128, or 256, got ", head_dim);

    auto opts_u8 = at::TensorOptions().dtype(at::kByte).device(x.device());
    auto opts_e8m0 = at::TensorOptions().dtype(at::kFloat8_e8m0fnu).device(x.device());
    at::Tensor x_packed = at::empty({batch, heads, seq_len, head_dim / 2}, opts_u8);
    at::Tensor x_scale = at::empty({batch, heads, seq_len, head_dim / 32}, opts_e8m0);

    mxfp4_quantise_permute_seq(
        x.data_ptr(), x_packed.data_ptr(), x_scale.data_ptr(),
        batch * heads, seq_len, head_dim, false, is_bf16);

    return {x_packed, x_scale};
}

static std::vector<at::Tensor> mxfp4_quantize_transposed(const at::Tensor& x, bool is_bf16 = false) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D [batch, heads, seq, head_dim]");

    const int batch = x.size(0);
    const int heads = x.size(1);
    const int seq_len = x.size(2);
    const int head_dim = x.size(3);
    TORCH_CHECK(head_dim == 64 || head_dim == 128 || head_dim == 256,
                "head_dim must be 64, 128, or 256, got ", head_dim);

    constexpr int seq_block = 128;
    const int padded_seq = ((seq_len + seq_block - 1) / seq_block) * seq_block;
    auto opts_u8 = at::TensorOptions().dtype(at::kByte).device(x.device());
    auto opts_e8m0 = at::TensorOptions().dtype(at::kFloat8_e8m0fnu).device(x.device());
    at::Tensor x_packed = at::empty({batch, heads, head_dim, padded_seq / 2}, opts_u8);
    at::Tensor x_scale = at::empty({batch, heads, head_dim, padded_seq / 32}, opts_e8m0);

    mxfp4_quantise_transpose(
        x.data_ptr(), x_packed.data_ptr(), x_scale.data_ptr(),
        batch * heads, seq_len, head_dim, is_bf16);

    return {x_packed, x_scale};
}

static std::vector<at::Tensor> mxfp4_quantize_transposed_permuted(const at::Tensor& x, bool is_bf16 = false) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D [batch, heads, seq, head_dim]");

    const int batch = x.size(0);
    const int heads = x.size(1);
    const int seq_len = x.size(2);
    const int head_dim = x.size(3);
    TORCH_CHECK(head_dim == 64 || head_dim == 128 || head_dim == 256,
                "head_dim must be 64, 128, or 256, got ", head_dim);

    constexpr int seq_block = 128;
    const int padded_seq = ((seq_len + seq_block - 1) / seq_block) * seq_block;
    auto opts_u8 = at::TensorOptions().dtype(at::kByte).device(x.device());
    auto opts_e8m0 = at::TensorOptions().dtype(at::kFloat8_e8m0fnu).device(x.device());
    at::Tensor x_packed = at::empty({batch, heads, head_dim, padded_seq / 2}, opts_u8);
    at::Tensor x_scale = at::empty({batch, heads, head_dim, padded_seq / 32}, opts_e8m0);

    mxfp4_quantise_transpose_permute_seq(
        x.data_ptr(), x_packed.data_ptr(), x_scale.data_ptr(),
        batch * heads, seq_len, head_dim, false, is_bf16);

    return {x_packed, x_scale};
}

static at::Tensor block_mean_topk_impl(
    const at::Tensor& q_mean,
    const at::Tensor& k_mean,
    int topk_count,
    bool causal = true,
    bool is_bf16 = false) {
    TORCH_CHECK(q_mean.is_cuda() && k_mean.is_cuda(),
                "q_mean and k_mean must be CUDA tensors");
    TORCH_CHECK(q_mean.is_contiguous() && k_mean.is_contiguous(),
                "q_mean and k_mean must be contiguous");
    TORCH_CHECK(q_mean.dim() == 4 && k_mean.dim() == 4,
                "q_mean and k_mean must be 4D tensors");
    TORCH_CHECK(q_mean.size(0) == k_mean.size(0),
                "q_mean and k_mean batch dimensions must match");
    TORCH_CHECK(q_mean.size(1) == k_mean.size(1),
                "q_mean and k_mean head dimensions must match");
    TORCH_CHECK(q_mean.size(3) == k_mean.size(3),
                "q_mean and k_mean head_dim must match");

    const int batch = q_mean.size(0);
    const int heads = q_mean.size(1);
    const int flat_heads = batch * heads;
    const int num_q_blocks = q_mean.size(2);
    const int num_kv_blocks = k_mean.size(2);
    const int head_dim = q_mean.size(3);
    TORCH_CHECK(head_dim == 64 || head_dim == 128 || head_dim == 256,
                "head_dim must be 64, 128, or 256, got ", head_dim);
    TORCH_CHECK(num_kv_blocks <= 2048,
                "block selector supports <= 2048 KV blocks, got ", num_kv_blocks);
    TORCH_CHECK(topk_count >= 0 && topk_count <= num_kv_blocks,
                "topk_count must be in [0, num_kv_blocks]");

    auto opts = at::TensorOptions().dtype(at::kInt).device(q_mean.device());
    at::Tensor topk = at::empty({flat_heads, num_q_blocks, topk_count}, opts);
    if (topk_count == 0) {
        return topk;
    }

    block_mean_topk(
        q_mean.data_ptr(), k_mean.data_ptr(), topk.data_ptr(),
        flat_heads, num_q_blocks, num_kv_blocks, head_dim, topk_count, causal, is_bf16);

    return topk;
}

static at::Tensor quest_block_topk_impl(
    const at::Tensor& q_mean,
    const at::Tensor& k_min,
    const at::Tensor& k_max,
    int topk_count,
    bool causal = true,
    bool is_bf16 = false) {
    TORCH_CHECK(q_mean.is_cuda() && k_min.is_cuda() && k_max.is_cuda(),
                "q_mean, k_min, and k_max must be CUDA tensors");
    TORCH_CHECK(q_mean.is_contiguous() && k_min.is_contiguous() && k_max.is_contiguous(),
                "q_mean, k_min, and k_max must be contiguous");
    TORCH_CHECK(q_mean.dim() == 4 && k_min.dim() == 4 && k_max.dim() == 4,
                "q_mean, k_min, and k_max must be 4D tensors");
    TORCH_CHECK(k_min.sizes() == k_max.sizes(),
                "k_min and k_max shapes must match");
    TORCH_CHECK(q_mean.size(0) == k_min.size(0),
                "q_mean and key stat batch dimensions must match");
    TORCH_CHECK(q_mean.size(1) == k_min.size(1),
                "q_mean and key stat head dimensions must match");
    TORCH_CHECK(q_mean.size(3) == k_min.size(3),
                "q_mean and key stat head_dim must match");

    const int batch = q_mean.size(0);
    const int heads = q_mean.size(1);
    const int flat_heads = batch * heads;
    const int num_q_blocks = q_mean.size(2);
    const int num_kv_blocks = k_min.size(2);
    const int head_dim = q_mean.size(3);
    TORCH_CHECK(head_dim == 64 || head_dim == 128 || head_dim == 256,
                "head_dim must be 64, 128, or 256, got ", head_dim);
    TORCH_CHECK(num_kv_blocks <= 2048,
                "QUEST block selector supports <= 2048 KV blocks, got ", num_kv_blocks);
    TORCH_CHECK(topk_count >= 0 && topk_count <= num_kv_blocks,
                "topk_count must be in [0, num_kv_blocks]");

    auto opts = at::TensorOptions().dtype(at::kInt).device(q_mean.device());
    at::Tensor topk = at::empty({flat_heads, num_q_blocks, topk_count}, opts);
    if (topk_count == 0) {
        return topk;
    }

    quest_block_topk(
        q_mean.data_ptr(), k_min.data_ptr(), k_max.data_ptr(), topk.data_ptr(),
        flat_heads, num_q_blocks, num_kv_blocks, head_dim, topk_count, causal, is_bf16);

    return topk;
}

static at::Tensor single_query_key_mean_topk_impl(
    const at::Tensor& q_grouped,
    const at::Tensor& k_mean,
    int topk_count,
    int num_kv_blocks,
    bool is_bf16 = false) {
    TORCH_CHECK(q_grouped.is_cuda() && k_mean.is_cuda(),
                "q_grouped and k_mean must be CUDA tensors");
    TORCH_CHECK(q_grouped.is_contiguous() && k_mean.is_contiguous(),
                "q_grouped and k_mean must be contiguous");
    TORCH_CHECK(q_grouped.dim() == 4 && k_mean.dim() == 4,
                "q_grouped and k_mean must be 4D tensors");
    TORCH_CHECK(q_grouped.size(0) == k_mean.size(0),
                "q_grouped and k_mean batch dimensions must match");
    TORCH_CHECK(q_grouped.size(1) == k_mean.size(1),
                "q_grouped and k_mean KV-head dimensions must match");
    TORCH_CHECK(q_grouped.size(3) == k_mean.size(3),
                "q_grouped and k_mean head_dim must match");

    const int batch = q_grouped.size(0);
    const int kv_heads = q_grouped.size(1);
    const int flat_heads = batch * kv_heads;
    const int groups = q_grouped.size(2);
    const int head_dim = q_grouped.size(3);
    const int k_mean_capacity_blocks = k_mean.size(2);
    TORCH_CHECK(groups >= 1 && groups <= 16,
                "grouped single-query selector expects groups in [1, 16], got ", groups);
    TORCH_CHECK(head_dim == 64 || head_dim == 128 || head_dim == 256,
                "head_dim must be 64, 128, or 256, got ", head_dim);
    TORCH_CHECK(num_kv_blocks >= 0 && num_kv_blocks <= k_mean_capacity_blocks,
                "num_kv_blocks must be in [0, k_mean capacity]");
    TORCH_CHECK(num_kv_blocks <= 2048,
                "single-query block selector supports <= 2048 KV blocks, got ", num_kv_blocks);
    TORCH_CHECK(topk_count >= 0 && topk_count <= num_kv_blocks,
                "topk_count must be in [0, num_kv_blocks]");

    auto opts = at::TensorOptions().dtype(at::kInt).device(q_grouped.device());
    at::Tensor topk = at::empty({flat_heads, topk_count}, opts);
    if (topk_count == 0) {
        return topk;
    }

    if (num_kv_blocks <= 128) {
        single_query_key_mean_topk(
            q_grouped.data_ptr(), k_mean.data_ptr(), topk.data_ptr(),
            flat_heads, groups, num_kv_blocks, k_mean_capacity_blocks,
            head_dim, topk_count, is_bf16);
        return topk;
    }

    constexpr int chunk_blocks = 64;
    const int chunk_count = (num_kv_blocks + chunk_blocks - 1) / chunk_blocks;
    const int local_count = std::min(topk_count, chunk_blocks);
    at::Tensor local_scores = at::empty(
        {flat_heads, chunk_count, local_count},
        at::TensorOptions().dtype(at::kFloat).device(q_grouped.device()));
    at::Tensor local_indices = at::empty({flat_heads, chunk_count, local_count}, opts);
    at::Tensor done_counts = at::empty({flat_heads}, opts);
    done_counts.zero_();
    single_query_key_mean_topk_chunked(
        q_grouped.data_ptr(), k_mean.data_ptr(), topk.data_ptr(),
        local_scores.data_ptr(), local_indices.data_ptr(), done_counts.data_ptr(),
        flat_heads, groups, num_kv_blocks, k_mean_capacity_blocks,
        head_dim, topk_count, chunk_count, local_count, is_bf16);

    return topk;
}

static at::Tensor single_query_quest_topk_impl(
    const at::Tensor& q_grouped,
    const at::Tensor& k_min,
    const at::Tensor& k_max,
    int topk_count,
    int num_kv_blocks,
    bool is_bf16 = false) {
    TORCH_CHECK(q_grouped.is_cuda() && k_min.is_cuda() && k_max.is_cuda(),
                "q_grouped, k_min, and k_max must be CUDA tensors");
    TORCH_CHECK(q_grouped.is_contiguous() && k_min.is_contiguous() && k_max.is_contiguous(),
                "q_grouped, k_min, and k_max must be contiguous");
    TORCH_CHECK(q_grouped.dim() == 4 && k_min.dim() == 4 && k_max.dim() == 4,
                "q_grouped, k_min, and k_max must be 4D tensors");
    TORCH_CHECK(k_min.sizes() == k_max.sizes(),
                "k_min and k_max shapes must match");
    TORCH_CHECK(q_grouped.size(0) == k_min.size(0),
                "q_grouped and key stat batch dimensions must match");
    TORCH_CHECK(q_grouped.size(1) == k_min.size(1),
                "q_grouped and key stat KV-head dimensions must match");
    TORCH_CHECK(q_grouped.size(3) == k_min.size(3),
                "q_grouped and key stat head_dim must match");

    const int batch = q_grouped.size(0);
    const int kv_heads = q_grouped.size(1);
    const int flat_heads = batch * kv_heads;
    const int groups = q_grouped.size(2);
    const int head_dim = q_grouped.size(3);
    const int k_stat_capacity_blocks = k_min.size(2);
    TORCH_CHECK(groups >= 1 && groups <= 16,
                "grouped single-query selector expects groups in [1, 16], got ", groups);
    TORCH_CHECK(head_dim == 64 || head_dim == 128 || head_dim == 256,
                "head_dim must be 64, 128, or 256, got ", head_dim);
    TORCH_CHECK(num_kv_blocks >= 0 && num_kv_blocks <= k_stat_capacity_blocks,
                "num_kv_blocks must be in [0, key stat capacity]");
    TORCH_CHECK(num_kv_blocks <= 2048,
                "QUEST single-query block selector supports <= 2048 KV blocks, got ", num_kv_blocks);
    TORCH_CHECK(topk_count >= 0 && topk_count <= num_kv_blocks,
                "topk_count must be in [0, num_kv_blocks]");

    auto opts = at::TensorOptions().dtype(at::kInt).device(q_grouped.device());
    at::Tensor topk = at::empty({flat_heads, topk_count}, opts);
    if (topk_count == 0) {
        return topk;
    }

    single_query_quest_topk(
        q_grouped.data_ptr(), k_min.data_ptr(), k_max.data_ptr(), topk.data_ptr(),
        flat_heads, groups, num_kv_blocks, k_stat_capacity_blocks,
        head_dim, topk_count, is_bf16);

    return topk;
}

static at::Tensor local_block_topk_impl(
    const at::Tensor& q,
    int num_kv_blocks,
    int topk_count,
    bool causal = true,
    int block_size = 64) {
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(q.dim() == 4, "q must be 4D [batch, heads, seq, head_dim]");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(q.size(2) % block_size == 0,
                "q sequence length must be divisible by block_size");
    TORCH_CHECK(num_kv_blocks >= 0, "num_kv_blocks must be non-negative");
    TORCH_CHECK(topk_count >= 0 && topk_count <= num_kv_blocks,
                "topk_count must be in [0, num_kv_blocks]");

    const int batch = q.size(0);
    const int heads = q.size(1);
    const int flat_heads = batch * heads;
    const int num_q_blocks = q.size(2) / block_size;

    auto opts = at::TensorOptions().dtype(at::kInt).device(q.device());
    at::Tensor topk = at::empty({flat_heads, num_q_blocks, topk_count}, opts);
    if (topk_count == 0 || flat_heads == 0 || num_q_blocks == 0) {
        return topk;
    }

    const cudaError_t err = local_block_topk(
        topk.data_ptr<int32_t>(), flat_heads, num_q_blocks, num_kv_blocks, topk_count, causal);
    TORCH_CHECK(err == cudaSuccess,
                "local_block_topk kernel launch failed: ", cudaGetErrorString(err));

    return topk;
}

static at::Tensor single_query_local_topk_impl(
    const at::Tensor& q_grouped,
    int topk_count,
    int num_kv_blocks) {
    TORCH_CHECK(q_grouped.is_cuda(), "q_grouped must be a CUDA tensor");
    TORCH_CHECK(q_grouped.dim() == 4,
                "q_grouped must be 4D [batch, kv_heads, groups, head_dim]");
    TORCH_CHECK(num_kv_blocks >= 0, "num_kv_blocks must be non-negative");
    TORCH_CHECK(topk_count >= 0 && topk_count <= num_kv_blocks,
                "topk_count must be in [0, num_kv_blocks]");

    const int batch = q_grouped.size(0);
    const int kv_heads = q_grouped.size(1);
    const int flat_heads = batch * kv_heads;

    auto opts = at::TensorOptions().dtype(at::kInt).device(q_grouped.device());
    at::Tensor topk = at::empty({flat_heads, topk_count}, opts);
    if (topk_count == 0 || flat_heads == 0) {
        return topk;
    }

    const cudaError_t err = single_query_local_topk(
        topk.data_ptr<int32_t>(), flat_heads, num_kv_blocks, topk_count);
    TORCH_CHECK(err == cudaSuccess,
                "single_query_local_topk kernel launch failed: ", cudaGetErrorString(err));

    return topk;
}

static at::Tensor single_query_key_mean_topk_into_impl(
    const at::Tensor& q_grouped,
    const at::Tensor& k_mean,
    at::Tensor& topk,
    at::Tensor& local_scores,
    at::Tensor& local_indices,
    at::Tensor& done_counts,
    int num_kv_blocks,
    bool is_bf16 = false) {
    TORCH_CHECK(q_grouped.is_cuda() && k_mean.is_cuda(),
                "q_grouped and k_mean must be CUDA tensors");
    TORCH_CHECK(topk.is_cuda() && local_scores.is_cuda() &&
                local_indices.is_cuda() && done_counts.is_cuda(),
                "top-k workspace tensors must be CUDA tensors");
    TORCH_CHECK(q_grouped.is_contiguous() && k_mean.is_contiguous() &&
                topk.is_contiguous() && local_scores.is_contiguous() &&
                local_indices.is_contiguous() && done_counts.is_contiguous(),
                "q_grouped, k_mean, and top-k workspace tensors must be contiguous");
    TORCH_CHECK(topk.dtype() == at::kInt && local_indices.dtype() == at::kInt &&
                done_counts.dtype() == at::kInt,
                "topk, local_indices, and done_counts must be int32");
    TORCH_CHECK(local_scores.dtype() == at::kFloat,
                "local_scores must be float32");
    TORCH_CHECK(q_grouped.dim() == 4 && k_mean.dim() == 4,
                "q_grouped and k_mean must be 4D tensors");
    TORCH_CHECK(q_grouped.size(0) == k_mean.size(0),
                "q_grouped and k_mean batch dimensions must match");
    TORCH_CHECK(q_grouped.size(1) == k_mean.size(1),
                "q_grouped and k_mean KV-head dimensions must match");
    TORCH_CHECK(q_grouped.size(3) == k_mean.size(3),
                "q_grouped and k_mean head_dim must match");

    const int batch = q_grouped.size(0);
    const int kv_heads = q_grouped.size(1);
    const int flat_heads = batch * kv_heads;
    const int groups = q_grouped.size(2);
    const int head_dim = q_grouped.size(3);
    const int k_mean_capacity_blocks = k_mean.size(2);
    const int topk_count = topk.size(1);
    TORCH_CHECK(topk.dim() == 2 && topk.size(0) == flat_heads,
                "topk must be shaped [batch * kv_heads, topk_count]");
    TORCH_CHECK(done_counts.dim() == 1 && done_counts.size(0) == flat_heads,
                "done_counts must be shaped [batch * kv_heads]");
    TORCH_CHECK(groups >= 1 && groups <= 16,
                "grouped single-query selector expects groups in [1, 16], got ", groups);
    TORCH_CHECK(head_dim == 64 || head_dim == 128 || head_dim == 256,
                "head_dim must be 64, 128, or 256, got ", head_dim);
    TORCH_CHECK(num_kv_blocks >= 0 && num_kv_blocks <= k_mean_capacity_blocks,
                "num_kv_blocks must be in [0, k_mean capacity]");
    TORCH_CHECK(num_kv_blocks <= 2048,
                "single-query block selector supports <= 2048 KV blocks, got ", num_kv_blocks);
    TORCH_CHECK(topk_count >= 0 && topk_count <= num_kv_blocks,
                "topk_count must be in [0, num_kv_blocks]");
    if (topk_count == 0) {
        return topk;
    }
    done_counts.zero_();

    if (num_kv_blocks <= 128) {
        single_query_key_mean_topk(
            q_grouped.data_ptr(), k_mean.data_ptr(), topk.data_ptr(),
            flat_heads, groups, num_kv_blocks, k_mean_capacity_blocks,
            head_dim, topk_count, is_bf16);
        return topk;
    }

    constexpr int chunk_blocks = 64;
    const int chunk_count = (num_kv_blocks + chunk_blocks - 1) / chunk_blocks;
    const int local_count = std::min(topk_count, chunk_blocks);
    TORCH_CHECK(local_scores.dim() == 3 && local_scores.size(0) == flat_heads &&
                local_scores.size(1) >= chunk_count && local_scores.size(2) >= local_count,
                "local_scores workspace is too small");
    TORCH_CHECK(local_indices.dim() == 3 && local_indices.size(0) == flat_heads &&
                local_indices.size(1) >= chunk_count && local_indices.size(2) >= local_count,
                "local_indices workspace is too small");
    single_query_key_mean_topk_chunked(
        q_grouped.data_ptr(), k_mean.data_ptr(), topk.data_ptr(),
        local_scores.data_ptr(), local_indices.data_ptr(), done_counts.data_ptr(),
        flat_heads, groups, num_kv_blocks, k_mean_capacity_blocks,
        head_dim, topk_count, chunk_count, local_count, is_bf16);

    return topk;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp4_attention_causal_nvfp4_packed", &fp4_attention_causal_nvfp4_packed,
          pybind11::arg("q_packed"),
          pybind11::arg("k_packed"),
          pybind11::arg("v_packed_t"),
          pybind11::arg("q_scale"),
          pybind11::arg("k_scale"),
          pybind11::arg("v_scale_t"),
          pybind11::arg("is_bf16") = false,
          "Pure NVFP4 causal attention over packed tensors");
    m.def("fp4_attention_noncausal_nvfp4_packed", &fp4_attention_noncausal_nvfp4_packed,
          pybind11::arg("q_packed"),
          pybind11::arg("k_packed"),
          pybind11::arg("v_packed_t"),
          pybind11::arg("q_scale"),
          pybind11::arg("k_scale"),
          pybind11::arg("v_scale_t"),
          pybind11::arg("is_bf16") = false,
          "Pure NVFP4 non-causal attention over packed tensors");
    m.def("fp4_attention_varlen_nvfp4_packed", &fp4_attention_varlen_nvfp4_packed,
          pybind11::arg("q_packed"),
          pybind11::arg("k_packed"),
          pybind11::arg("v_packed_t"),
          pybind11::arg("q_scale"),
          pybind11::arg("k_scale"),
          pybind11::arg("v_scale_t"),
          pybind11::arg("cu_seqlens"),
          pybind11::arg("cu_seqlens_padded"),
          pybind11::arg("max_seqlen"),
          pybind11::arg("total_tokens"),
          pybind11::arg("causal"),
          pybind11::arg("is_bf16") = false,
          pybind11::arg("key_bias") = pybind11::none(),
          "Pure NVFP4 variable-length self-attention over sequences packed along the token axis");
    m.def("fp4_attention_varlen_nvfp4_packed_fp4_out", &fp4_attention_varlen_nvfp4_packed_fp4_out,
          pybind11::arg("q_packed"),
          pybind11::arg("k_packed"),
          pybind11::arg("v_packed_t"),
          pybind11::arg("q_scale"),
          pybind11::arg("k_scale"),
          pybind11::arg("v_scale_t"),
          pybind11::arg("cu_seqlens"),
          pybind11::arg("cu_seqlens_padded"),
          pybind11::arg("max_seqlen"),
          pybind11::arg("total_tokens"),
          pybind11::arg("causal"),
          pybind11::arg("is_bf16"),
          pybind11::arg("key_bias"),
          pybind11::arg("out_global_scale"),
          "Variable-length NVFP4 self-attention whose output is the NVFP4 input of the next Linear "
          "(packed values and swizzled scales)");
    m.def("fp4_attention_varlen_nvfp4_cached", &fp4_attention_varlen_nvfp4_cached,
          pybind11::arg("q_packed"),
          pybind11::arg("k_packed"),
          pybind11::arg("v_packed_t"),
          pybind11::arg("q_scale"),
          pybind11::arg("k_scale"),
          pybind11::arg("v_scale_t"),
          pybind11::arg("cu_seqlens"),
          pybind11::arg("cu_seqlens_padded"),
          pybind11::arg("max_seqlen"),
          pybind11::arg("out_tokens"),
          pybind11::arg("causal"),
          pybind11::arg("is_bf16"),
          pybind11::arg("key_bias"),
          pybind11::arg("q_start"),
          pybind11::arg("out_start"),
          pybind11::kw_only(),
          pybind11::arg("out_global_scale") = pybind11::none(),
          pybind11::arg("return_lse") = false,
          pybind11::arg("out") = pybind11::none(),
          pybind11::arg("out_scales") = pybind11::none(),
          pybind11::arg("lse") = pybind11::none(),
          pybind11::arg("q_bf16") = pybind11::none(),
          pybind11::arg("key_cache") = pybind11::none(),
          pybind11::arg("value_cache") = pybind11::none(),
          pybind11::arg("block_table") = pybind11::none(),
          pybind11::arg("prefix_len") = pybind11::none(),
          pybind11::arg("key_mean") = pybind11::none(),
          "Variable-length NVFP4 attention for the query rows from q_start on (keys before them come from a "
          "cache), optionally with NVFP4 output and the log-sum-exp of each row; with prefix_len, keys "
          "0 .. prefix_len[seq] - 1 of each sequence are attended in bfloat16 from a paged cache after its FP4 keys");
    m.def("fp4_attention_single_query_nvfp4_packed", &fp4_attention_single_query_nvfp4_packed,
          pybind11::arg("q_packed"),
          pybind11::arg("k_packed"),
          pybind11::arg("v_packed_t"),
          pybind11::arg("q_scale"),
          pybind11::arg("k_scale"),
          pybind11::arg("v_scale_t"),
          pybind11::arg("is_bf16") = false,
          "Pure NVFP4 single-query attention over packed tensors");
    m.def("fp4_attention_causal_mxfp4_packed", &fp4_attention_causal_mxfp4_packed,
          pybind11::arg("q_packed"),
          pybind11::arg("k_packed"),
          pybind11::arg("v_packed_t"),
          pybind11::arg("q_scale"),
          pybind11::arg("k_scale"),
          pybind11::arg("v_scale_t"),
          pybind11::arg("is_bf16") = false,
          "Pure MXFP4 causal attention over packed tensors");
    m.def("fp4_attention_noncausal_mxfp4_packed", &fp4_attention_noncausal_mxfp4_packed,
          pybind11::arg("q_packed"),
          pybind11::arg("k_packed"),
          pybind11::arg("v_packed_t"),
          pybind11::arg("q_scale"),
          pybind11::arg("k_scale"),
          pybind11::arg("v_scale_t"),
          pybind11::arg("is_bf16") = false,
          "Pure MXFP4 non-causal attention over packed tensors");
    m.def("fp4_attention_single_query_mxfp4_packed", &fp4_attention_single_query_mxfp4_packed,
          pybind11::arg("q_packed"),
          pybind11::arg("k_packed"),
          pybind11::arg("v_packed_t"),
          pybind11::arg("q_scale"),
          pybind11::arg("k_scale"),
          pybind11::arg("v_scale_t"),
          pybind11::arg("is_bf16") = false,
          "Pure MXFP4 single-query attention over packed tensors");
    m.def("thrift_attention_causal_nvfp4_packed", &thrift_attention_causal_nvfp4_packed,
          pybind11::arg("q_hi"),
          pybind11::arg("k_hi"),
          pybind11::arg("v_hi"),
          pybind11::arg("selected_blocks"),
          pybind11::arg("q_packed"),
          pybind11::arg("k_packed"),
          pybind11::arg("v_packed_t"),
          pybind11::arg("q_scale"),
          pybind11::arg("k_scale"),
          pybind11::arg("v_scale_t"),
          pybind11::arg("is_bf16") = false,
          "ThriftAttention causal attention over packed tensors");
    m.def("thrift_attention_noncausal_nvfp4_packed", &thrift_attention_noncausal_nvfp4_packed,
          pybind11::arg("q_hi"),
          pybind11::arg("k_hi"),
          pybind11::arg("v_hi"),
          pybind11::arg("selected_blocks"),
          pybind11::arg("q_packed"),
          pybind11::arg("k_packed"),
          pybind11::arg("v_packed_t"),
          pybind11::arg("q_scale"),
          pybind11::arg("k_scale"),
          pybind11::arg("v_scale_t"),
          pybind11::arg("is_bf16") = false,
          "ThriftAttention non-causal attention over packed tensors");
    m.def("thrift_attention_single_query_nvfp4_packed", &thrift_attention_single_query_nvfp4_packed,
          pybind11::arg("q_hi"),
          pybind11::arg("k_hi"),
          pybind11::arg("v_hi"),
          pybind11::arg("selected_blocks"),
          pybind11::arg("q_packed"),
          pybind11::arg("k_packed"),
          pybind11::arg("v_packed_t"),
          pybind11::arg("q_scale"),
          pybind11::arg("k_scale"),
          pybind11::arg("v_scale_t"),
          pybind11::arg("is_bf16") = false,
          "ThriftAttention single-query attention over packed tensors");
    m.def("thrift_attention_single_query_mxfp4_packed", &thrift_attention_single_query_mxfp4_packed,
          pybind11::arg("q_hi"),
          pybind11::arg("k_hi"),
          pybind11::arg("v_hi"),
          pybind11::arg("selected_blocks"),
          pybind11::arg("q_packed"),
          pybind11::arg("k_packed"),
          pybind11::arg("v_packed_t"),
          pybind11::arg("q_scale"),
          pybind11::arg("k_scale"),
          pybind11::arg("v_scale_t"),
          pybind11::arg("is_bf16") = false,
          "ThriftAttention single-query attention over MXFP4 packed tensors");
    m.def("thrift_attention_causal_mxfp4_packed", &thrift_attention_causal_mxfp4_packed,
          pybind11::arg("q_hi"),
          pybind11::arg("k_hi"),
          pybind11::arg("v_hi"),
          pybind11::arg("selected_blocks"),
          pybind11::arg("q_packed"),
          pybind11::arg("k_packed"),
          pybind11::arg("v_packed_t"),
          pybind11::arg("q_scale"),
          pybind11::arg("k_scale"),
          pybind11::arg("v_scale_t"),
          pybind11::arg("is_bf16") = false,
          "ThriftAttention causal attention over MXFP4 packed tensors");
    m.def("thrift_attention_noncausal_mxfp4_packed", &thrift_attention_noncausal_mxfp4_packed,
          pybind11::arg("q_hi"),
          pybind11::arg("k_hi"),
          pybind11::arg("v_hi"),
          pybind11::arg("selected_blocks"),
          pybind11::arg("q_packed"),
          pybind11::arg("k_packed"),
          pybind11::arg("v_packed_t"),
          pybind11::arg("q_scale"),
          pybind11::arg("k_scale"),
          pybind11::arg("v_scale_t"),
          pybind11::arg("is_bf16") = false,
          "ThriftAttention non-causal attention over MXFP4 packed tensors");
    m.def("nvfp4_quantize", &nvfp4_quantize,
          pybind11::arg("x"),
          pybind11::arg("is_bf16") = false,
          "Quantize contiguous FP16/BF16 tensors to packed NVFP4 with FP8 scales");
    m.def("nvfp4_quantize_permuted", &nvfp4_quantize_permuted,
          pybind11::arg("x"),
          pybind11::arg("is_bf16") = false,
          "Quantize contiguous FP16/BF16 tensors to packed NVFP4 with Sage-style sequence permutation");
    m.def("nvfp4_quantize_transposed", &nvfp4_quantize_transposed,
          pybind11::arg("x"),
          pybind11::arg("is_bf16") = false,
          "Quantize contiguous FP16/BF16 tensors to transposed packed NVFP4 with FP8 scales");
    m.def("nvfp4_quantize_transposed_permuted", &nvfp4_quantize_transposed_permuted,
          pybind11::arg("x"),
          pybind11::arg("is_bf16") = false,
          "Quantize contiguous FP16/BF16 tensors to transposed packed NVFP4 with Sage-style sequence permutation");
    m.def("mxfp4_quantize", &mxfp4_quantize,
          pybind11::arg("x"),
          pybind11::arg("is_bf16") = false,
          "Quantize contiguous FP16/BF16 tensors to packed MXFP4 with E8M0 scales");
    m.def("mxfp4_quantize_permuted", &mxfp4_quantize_permuted,
          pybind11::arg("x"),
          pybind11::arg("is_bf16") = false,
          "Quantize contiguous FP16/BF16 tensors to packed MXFP4 with Sage-style sequence permutation");
    m.def("mxfp4_quantize_transposed", &mxfp4_quantize_transposed,
          pybind11::arg("x"),
          pybind11::arg("is_bf16") = false,
          "Quantize contiguous FP16/BF16 tensors to transposed packed MXFP4 with E8M0 scales");
    m.def("mxfp4_quantize_transposed_permuted", &mxfp4_quantize_transposed_permuted,
          pybind11::arg("x"),
          pybind11::arg("is_bf16") = false,
          "Quantize contiguous FP16/BF16 tensors to transposed packed MXFP4 with Sage-style sequence permutation");
    m.def("block_mean_topk", &block_mean_topk_impl,
          pybind11::arg("q_mean"),
          pybind11::arg("k_mean"),
          pybind11::arg("topk_count"),
          pybind11::arg("causal") = true,
          pybind11::arg("is_bf16") = false,
          "Select KV blocks using block-mean QK scores");
    m.def("quest_block_topk", &quest_block_topk_impl,
          pybind11::arg("q_mean"),
          pybind11::arg("k_min"),
          pybind11::arg("k_max"),
          pybind11::arg("topk_count"),
          pybind11::arg("causal") = true,
          pybind11::arg("is_bf16") = false,
          "Select KV blocks using QUEST query-aware min/max scores");
    m.def("single_query_key_mean_topk", &single_query_key_mean_topk_impl,
          pybind11::arg("q_grouped"),
          pybind11::arg("k_mean"),
          pybind11::arg("topk_count"),
          pybind11::arg("num_kv_blocks"),
          pybind11::arg("is_bf16") = false,
          "Select decode KV blocks using grouped single-query block-mean QK scores");
    m.def("single_query_quest_topk", &single_query_quest_topk_impl,
          pybind11::arg("q_grouped"),
          pybind11::arg("k_min"),
          pybind11::arg("k_max"),
          pybind11::arg("topk_count"),
          pybind11::arg("num_kv_blocks"),
          pybind11::arg("is_bf16") = false,
          "Select decode KV blocks using grouped single-query QUEST min/max scores");
    m.def("local_block_topk", &local_block_topk_impl,
          pybind11::arg("q"),
          pybind11::arg("num_kv_blocks"),
          pybind11::arg("topk_count"),
          pybind11::arg("causal") = true,
          pybind11::arg("block_size") = 64,
          "Select a contiguous local KV-block window for each query block");
    m.def("single_query_local_topk", &single_query_local_topk_impl,
          pybind11::arg("q_grouped"),
          pybind11::arg("topk_count"),
          pybind11::arg("num_kv_blocks"),
          "Select the trailing local KV-block window for decode");
    m.def("single_query_key_mean_topk_into", &single_query_key_mean_topk_into_impl,
          pybind11::arg("q_grouped"),
          pybind11::arg("k_mean"),
          pybind11::arg("topk"),
          pybind11::arg("local_scores"),
          pybind11::arg("local_indices"),
          pybind11::arg("done_counts"),
          pybind11::arg("num_kv_blocks"),
          pybind11::arg("is_bf16") = false,
          "Select decode KV blocks into preallocated top-k workspace");
}
