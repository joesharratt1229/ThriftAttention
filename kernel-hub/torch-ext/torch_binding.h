#pragma once

#include <ATen/ATen.h>
#include <vector>

at::Tensor fp4_attention_causal_nvfp4_packed(
    const at::Tensor& q_packed, const at::Tensor& k_packed, const at::Tensor& v_packed_t,
    const at::Tensor& q_scale, const at::Tensor& k_scale, const at::Tensor& v_scale_t,
    bool is_bf16);

at::Tensor fp4_attention_noncausal_nvfp4_packed(
    const at::Tensor& q_packed, const at::Tensor& k_packed, const at::Tensor& v_packed_t,
    const at::Tensor& q_scale, const at::Tensor& k_scale, const at::Tensor& v_scale_t,
    bool is_bf16);

at::Tensor fp4_attention_single_query_nvfp4_packed(
    const at::Tensor& q_packed, const at::Tensor& k_packed, const at::Tensor& v_packed_t,
    const at::Tensor& q_scale, const at::Tensor& k_scale, const at::Tensor& v_scale_t,
    bool is_bf16);

at::Tensor fp4_attention_causal_mxfp4_packed(
    const at::Tensor& q_packed, const at::Tensor& k_packed, const at::Tensor& v_packed_t,
    const at::Tensor& q_scale, const at::Tensor& k_scale, const at::Tensor& v_scale_t,
    bool is_bf16);

at::Tensor fp4_attention_noncausal_mxfp4_packed(
    const at::Tensor& q_packed, const at::Tensor& k_packed, const at::Tensor& v_packed_t,
    const at::Tensor& q_scale, const at::Tensor& k_scale, const at::Tensor& v_scale_t,
    bool is_bf16);

at::Tensor fp4_attention_single_query_mxfp4_packed(
    const at::Tensor& q_packed, const at::Tensor& k_packed, const at::Tensor& v_packed_t,
    const at::Tensor& q_scale, const at::Tensor& k_scale, const at::Tensor& v_scale_t,
    bool is_bf16);

at::Tensor thrift_attention_causal_nvfp4_packed(
    const at::Tensor& q_hi, const at::Tensor& k_hi, const at::Tensor& v_hi,
    const at::Tensor& selected_blocks,
    const at::Tensor& q_packed, const at::Tensor& k_packed, const at::Tensor& v_packed_t,
    const at::Tensor& q_scale, const at::Tensor& k_scale, const at::Tensor& v_scale_t,
    bool is_bf16);

at::Tensor thrift_attention_noncausal_nvfp4_packed(
    const at::Tensor& q_hi, const at::Tensor& k_hi, const at::Tensor& v_hi,
    const at::Tensor& selected_blocks,
    const at::Tensor& q_packed, const at::Tensor& k_packed, const at::Tensor& v_packed_t,
    const at::Tensor& q_scale, const at::Tensor& k_scale, const at::Tensor& v_scale_t,
    bool is_bf16);

at::Tensor thrift_attention_single_query_nvfp4_packed(
    const at::Tensor& q_hi, const at::Tensor& k_hi, const at::Tensor& v_hi,
    const at::Tensor& selected_blocks,
    const at::Tensor& q_packed, const at::Tensor& k_packed, const at::Tensor& v_packed_t,
    const at::Tensor& q_scale, const at::Tensor& k_scale, const at::Tensor& v_scale_t,
    bool is_bf16);

at::Tensor thrift_attention_causal_mxfp4_packed(
    const at::Tensor& q_hi, const at::Tensor& k_hi, const at::Tensor& v_hi,
    const at::Tensor& selected_blocks,
    const at::Tensor& q_packed, const at::Tensor& k_packed, const at::Tensor& v_packed_t,
    const at::Tensor& q_scale, const at::Tensor& k_scale, const at::Tensor& v_scale_t,
    bool is_bf16);

at::Tensor thrift_attention_noncausal_mxfp4_packed(
    const at::Tensor& q_hi, const at::Tensor& k_hi, const at::Tensor& v_hi,
    const at::Tensor& selected_blocks,
    const at::Tensor& q_packed, const at::Tensor& k_packed, const at::Tensor& v_packed_t,
    const at::Tensor& q_scale, const at::Tensor& k_scale, const at::Tensor& v_scale_t,
    bool is_bf16);

at::Tensor thrift_attention_single_query_mxfp4_packed(
    const at::Tensor& q_hi, const at::Tensor& k_hi, const at::Tensor& v_hi,
    const at::Tensor& selected_blocks,
    const at::Tensor& q_packed, const at::Tensor& k_packed, const at::Tensor& v_packed_t,
    const at::Tensor& q_scale, const at::Tensor& k_scale, const at::Tensor& v_scale_t,
    bool is_bf16);

std::vector<at::Tensor> nvfp4_quantize(const at::Tensor& x, bool is_bf16);
std::vector<at::Tensor> nvfp4_quantize_permuted(const at::Tensor& x, bool is_bf16);
std::vector<at::Tensor> nvfp4_quantize_transposed(const at::Tensor& x, bool is_bf16);
std::vector<at::Tensor> mxfp4_quantize(const at::Tensor& x, bool is_bf16);
std::vector<at::Tensor> mxfp4_quantize_permuted(const at::Tensor& x, bool is_bf16);
std::vector<at::Tensor> mxfp4_quantize_transposed(const at::Tensor& x, bool is_bf16);

// Block-selection wrappers use *_impl names to avoid clashing with the
// CUDA kernels they call (block_mean_topk, quest_block_topk, etc.).
// The public TORCH_LIBRARY op names match the kernel names.
at::Tensor block_mean_topk_impl(
    const at::Tensor& q_mean, const at::Tensor& k_mean,
    int64_t topk_count, bool causal, bool is_bf16);

at::Tensor quest_block_topk_impl(
    const at::Tensor& q_mean, const at::Tensor& k_min, const at::Tensor& k_max,
    int64_t topk_count, bool causal, bool is_bf16);

at::Tensor single_query_key_mean_topk_impl(
    const at::Tensor& q_grouped, const at::Tensor& k_mean,
    int64_t topk_count, int64_t num_kv_blocks, bool is_bf16);

at::Tensor single_query_quest_topk_impl(
    const at::Tensor& q_grouped, const at::Tensor& k_min, const at::Tensor& k_max,
    int64_t topk_count, int64_t num_kv_blocks, bool is_bf16);

at::Tensor single_query_key_mean_topk_into_impl(
    const at::Tensor& q_grouped, const at::Tensor& k_mean,
    at::Tensor& topk, at::Tensor& local_scores, at::Tensor& local_indices, at::Tensor& done_counts,
    int64_t num_kv_blocks, bool is_bf16);
