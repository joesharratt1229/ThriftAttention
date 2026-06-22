#include <cstdint>

#include <cuda_runtime.h>

namespace {

// Per-query-block local window selection.
//
// out: [flat_heads, num_q_blocks, topk_count]
__global__ __launch_bounds__(256)
void local_block_topk_kernel(
    int32_t* __restrict__ topk_out,
    int num_q_blocks,
    int num_kv_blocks,
    int topk_count,
    bool causal) {
    const int q_block_id = blockIdx.x;
    const int row_id = blockIdx.y * num_q_blocks + q_block_id;
    const int tid = threadIdx.x;
    int32_t* out = topk_out + static_cast<int64_t>(row_id) * topk_count;

    int start = 0;
    int valid_count = topk_count;
    if (causal) {
        const int end = min(q_block_id, num_kv_blocks - 1);
        const int available = min(end + 1, topk_count);
        start = max(0, end - topk_count + 1);
        valid_count = available;
    } else {
        const int center = min(q_block_id, num_kv_blocks - 1);
        const int max_start = max(num_kv_blocks - topk_count, 0);
        start = min(max(center - topk_count / 2, 0), max_start);
    }

    for (int rank = tid; rank < topk_count; rank += blockDim.x) {
        out[rank] = rank < valid_count ? start + rank : -1;
    }
}

// Single-token decode local window selection.
//
// out: [flat_heads, topk_count]
__global__ __launch_bounds__(256)
void single_query_local_topk_kernel(
    int32_t* __restrict__ topk_out,
    int num_kv_blocks,
    int topk_count) {
    const int flat_head_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int start = num_kv_blocks - topk_count;
    int32_t* out = topk_out + static_cast<int64_t>(flat_head_id) * topk_count;

    for (int rank = tid; rank < topk_count; rank += blockDim.x) {
        out[rank] = start + rank;
    }
}

}  // namespace

cudaError_t local_block_topk(
    int32_t* topk_out,
    int flat_heads,
    int num_q_blocks,
    int num_kv_blocks,
    int topk_count,
    bool causal) {
    if (flat_heads <= 0 || num_q_blocks <= 0 || topk_count <= 0) {
        return cudaSuccess;
    }
    const dim3 grid_size(num_q_blocks, flat_heads);
    local_block_topk_kernel<<<grid_size, 256>>>(
        topk_out, num_q_blocks, num_kv_blocks, topk_count, causal);
    return cudaGetLastError();
}

cudaError_t single_query_local_topk(
    int32_t* topk_out,
    int flat_heads,
    int num_kv_blocks,
    int topk_count) {
    if (flat_heads <= 0 || topk_count <= 0) {
        return cudaSuccess;
    }
    single_query_local_topk_kernel<<<flat_heads, 256>>>(topk_out, num_kv_blocks, topk_count);
    return cudaGetLastError();
}
