#include <cstdint>
#include <cstdio>
#include <float.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "thriftattention/sm120/cuda_common.cuh"

namespace {

// Per-query-block selection.
//
// q_mean: [flat_heads, num_q_blocks, head_dim]
// k_mean: [flat_heads, num_kv_blocks, head_dim]
// out:    [flat_heads, num_q_blocks, topk_count]
template<bool CAUSAL, int HEAD_DIM, int MAX_KV_BLOCKS>
__global__ __launch_bounds__(TA_WARP_SIZE)
void block_mean_topk_kernel(
    const half* __restrict__ q_mean,
    const half* __restrict__ k_mean,
    int32_t* __restrict__ topk_out,
    int num_q_blocks,
    int num_kv_blocks,
    int topk_count) {
    constexpr int ITEMS_PER_LANE = MAX_KV_BLOCKS / TA_WARP_SIZE;
    constexpr int ELEMS_PER_LANE = HEAD_DIM / TA_WARP_SIZE;

    const int bid = blockIdx.x;
    const int flat_head_id = bid / num_q_blocks;
    const int q_block_id = bid % num_q_blocks;
    const int lane_id = threadIdx.x;

    const half* q_row =
        q_mean + (static_cast<int64_t>(flat_head_id) * num_q_blocks + q_block_id) * HEAD_DIM;
    float q_reg[ELEMS_PER_LANE];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_LANE; i++) {
        q_reg[i] = __half2float(q_row[lane_id + i * TA_WARP_SIZE]);
    }

    const half* k_head = k_mean + static_cast<int64_t>(flat_head_id) * num_kv_blocks * HEAD_DIM;
    extern __shared__ uint8_t smem[];
    float* block_scores = reinterpret_cast<float*>(smem);

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        const half* k_row = k_head + kv_block * HEAD_DIM;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_LANE; i++) {
            dot += q_reg[i] * __half2float(k_row[lane_id + i * TA_WARP_SIZE]);
        }

        #pragma unroll
        for (int delta = 16; delta >= 1; delta >>= 1) {
            dot += __shfl_xor_sync(0xFFFFFFFF, dot, delta);
        }

        if (lane_id == 0) {
            block_scores[kv_block] = dot;
        }
    }
    __syncwarp();

    float values[ITEMS_PER_LANE];
    int indices[ITEMS_PER_LANE];
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_LANE; i++) {
        const int idx = i * TA_WARP_SIZE + lane_id;
        indices[i] = idx;
        values[i] = (idx < num_kv_blocks && (!CAUSAL || idx <= q_block_id))
            ? block_scores[idx]
            : -FLT_MAX;
    }

    int32_t* out =
        topk_out + (static_cast<int64_t>(flat_head_id) * num_q_blocks + q_block_id) * topk_count;

    for (int k = 0; k < topk_count; k++) {
        float best = values[0];
        int best_idx = indices[0];
        #pragma unroll
        for (int i = 1; i < ITEMS_PER_LANE; i++) {
            if (values[i] > best) {
                best = values[i];
                best_idx = indices[i];
            }
        }

        #pragma unroll
        for (int delta = 16; delta >= 1; delta >>= 1) {
            const float other = __shfl_xor_sync(0xFFFFFFFF, best, delta);
            const int other_idx = __shfl_xor_sync(0xFFFFFFFF, best_idx, delta);
            if (other > best) {
                best = other;
                best_idx = other_idx;
            }
        }

        const bool found = best > -FLT_MAX * 0.5f;
        if (lane_id == 0) out[k] = found ? best_idx : -1;
        best_idx = __shfl_sync(0xFFFFFFFF, best_idx, 0);

        #pragma unroll
        for (int i = 0; i < ITEMS_PER_LANE; i++) {
            if (found && indices[i] == best_idx) values[i] = -FLT_MAX;
        }
    }
}

// Single-token decode selection for grouped-query attention.
//
// q_grouped: [flat_kv_heads, groups, head_dim]
// k_mean:    [flat_kv_heads, k_mean_capacity_blocks, head_dim]
// out:       [flat_kv_heads, topk_count]
//
// Scores are max_g dot(q[flat_head, g], k_mean[flat_head, block]), matching
// the Python grouped decode selector.
template<int HEAD_DIM, int MAX_KV_BLOCKS>
__global__ __launch_bounds__(TA_WARP_SIZE)
void single_query_key_mean_topk_kernel(
    const half* __restrict__ q_grouped,
    const half* __restrict__ k_mean,
    int32_t* __restrict__ topk_out,
    int groups,
    int num_kv_blocks,
    int k_mean_capacity_blocks,
    int topk_count) {
    constexpr int ITEMS_PER_LANE = MAX_KV_BLOCKS / TA_WARP_SIZE;
    constexpr int ELEMS_PER_LANE = HEAD_DIM / TA_WARP_SIZE;

    const int flat_head_id = blockIdx.x;
    const int lane_id = threadIdx.x;

    const half* q_head =
        q_grouped + static_cast<int64_t>(flat_head_id) * groups * HEAD_DIM;
    const half* k_head =
        k_mean + static_cast<int64_t>(flat_head_id) * k_mean_capacity_blocks * HEAD_DIM;

    extern __shared__ uint8_t smem[];
    float* block_scores = reinterpret_cast<float*>(smem);

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        const half* k_row = k_head + kv_block * HEAD_DIM;
        float best = -FLT_MAX;

        for (int group_id = 0; group_id < groups; group_id++) {
            const half* q_row = q_head + group_id * HEAD_DIM;
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < ELEMS_PER_LANE; i++) {
                const int elem = lane_id + i * TA_WARP_SIZE;
                dot += __half2float(q_row[elem]) * __half2float(k_row[elem]);
            }

            #pragma unroll
            for (int delta = 16; delta >= 1; delta >>= 1) {
                dot += __shfl_xor_sync(0xFFFFFFFF, dot, delta);
            }

            if (lane_id == 0 && dot > best) {
                best = dot;
            }
        }

        if (lane_id == 0) {
            block_scores[kv_block] = best;
        }
    }
    __syncwarp();

    float values[ITEMS_PER_LANE];
    int indices[ITEMS_PER_LANE];
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_LANE; i++) {
        const int idx = i * TA_WARP_SIZE + lane_id;
        indices[i] = idx;
        values[i] = (idx < num_kv_blocks) ? block_scores[idx] : -FLT_MAX;
    }

    int32_t* out = topk_out + static_cast<int64_t>(flat_head_id) * topk_count;

    for (int k = 0; k < topk_count; k++) {
        float best = values[0];
        int best_idx = indices[0];
        #pragma unroll
        for (int i = 1; i < ITEMS_PER_LANE; i++) {
            if (values[i] > best) {
                best = values[i];
                best_idx = indices[i];
            }
        }

        #pragma unroll
        for (int delta = 16; delta >= 1; delta >>= 1) {
            const float other = __shfl_xor_sync(0xFFFFFFFF, best, delta);
            const int other_idx = __shfl_xor_sync(0xFFFFFFFF, best_idx, delta);
            if (other > best) {
                best = other;
                best_idx = other_idx;
            }
        }

        if (lane_id == 0) {
            out[k] = best_idx;
        }
        best_idx = __shfl_sync(0xFFFFFFFF, best_idx, 0);

        #pragma unroll
        for (int i = 0; i < ITEMS_PER_LANE; i++) {
            if (indices[i] == best_idx) values[i] = -FLT_MAX;
        }
    }
}

template<bool CAUSAL, int HEAD_DIM, int MAX_KV_BLOCKS>
void launch_block_mean_topk(
    const half* q_mean,
    const half* k_mean,
    int32_t* topk_out,
    int flat_heads,
    int num_q_blocks,
    int num_kv_blocks,
    int topk_count) {
    const int grid_size = flat_heads * num_q_blocks;
    const int smem_bytes = num_kv_blocks * static_cast<int>(sizeof(float));

    block_mean_topk_kernel<CAUSAL, HEAD_DIM, MAX_KV_BLOCKS>
        <<<grid_size, TA_WARP_SIZE, smem_bytes>>>(
            q_mean, k_mean, topk_out, num_q_blocks, num_kv_blocks, topk_count);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "block_mean_topk kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

template<bool CAUSAL, int HEAD_DIM>
void dispatch_block_mean_topk(
    const half* q_mean,
    const half* k_mean,
    int32_t* topk_out,
    int flat_heads,
    int num_q_blocks,
    int num_kv_blocks,
    int topk_count) {
    if (num_kv_blocks <= 128) {
        launch_block_mean_topk<CAUSAL, HEAD_DIM, 128>(
            q_mean, k_mean, topk_out, flat_heads, num_q_blocks, num_kv_blocks, topk_count);
    } else if (num_kv_blocks <= 512) {
        launch_block_mean_topk<CAUSAL, HEAD_DIM, 512>(
            q_mean, k_mean, topk_out, flat_heads, num_q_blocks, num_kv_blocks, topk_count);
    } else if (num_kv_blocks <= 1024) {
        launch_block_mean_topk<CAUSAL, HEAD_DIM, 1024>(
            q_mean, k_mean, topk_out, flat_heads, num_q_blocks, num_kv_blocks, topk_count);
    } else {
        launch_block_mean_topk<CAUSAL, HEAD_DIM, 2048>(
            q_mean, k_mean, topk_out, flat_heads, num_q_blocks, num_kv_blocks, topk_count);
    }
}

template<int HEAD_DIM, int MAX_KV_BLOCKS>
void launch_single_query_key_mean_topk(
    const half* q_grouped,
    const half* k_mean,
    int32_t* topk_out,
    int flat_heads,
    int groups,
    int num_kv_blocks,
    int k_mean_capacity_blocks,
    int topk_count) {
    const int smem_bytes = num_kv_blocks * static_cast<int>(sizeof(float));

    single_query_key_mean_topk_kernel<HEAD_DIM, MAX_KV_BLOCKS>
        <<<flat_heads, TA_WARP_SIZE, smem_bytes>>>(
            q_grouped, k_mean, topk_out, groups, num_kv_blocks,
            k_mean_capacity_blocks, topk_count);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "single_query_key_mean_topk kernel launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

template<int HEAD_DIM>
void dispatch_single_query_key_mean_topk(
    const half* q_grouped,
    const half* k_mean,
    int32_t* topk_out,
    int flat_heads,
    int groups,
    int num_kv_blocks,
    int k_mean_capacity_blocks,
    int topk_count) {
    if (num_kv_blocks <= 128) {
        launch_single_query_key_mean_topk<HEAD_DIM, 128>(
            q_grouped, k_mean, topk_out, flat_heads, groups, num_kv_blocks,
            k_mean_capacity_blocks, topk_count);
    } else if (num_kv_blocks <= 512) {
        launch_single_query_key_mean_topk<HEAD_DIM, 512>(
            q_grouped, k_mean, topk_out, flat_heads, groups, num_kv_blocks,
            k_mean_capacity_blocks, topk_count);
    } else if (num_kv_blocks <= 1024) {
        launch_single_query_key_mean_topk<HEAD_DIM, 1024>(
            q_grouped, k_mean, topk_out, flat_heads, groups, num_kv_blocks,
            k_mean_capacity_blocks, topk_count);
    } else {
        launch_single_query_key_mean_topk<HEAD_DIM, 2048>(
            q_grouped, k_mean, topk_out, flat_heads, groups, num_kv_blocks,
            k_mean_capacity_blocks, topk_count);
    }
}

}  // namespace

void block_mean_topk(
    const void* q_mean_raw,
    const void* k_mean_raw,
    void* topk_out_raw,
    int flat_heads,
    int num_q_blocks,
    int num_kv_blocks,
    int head_dim,
    int topk_count,
    bool causal) {
    const half* q_mean = reinterpret_cast<const half*>(q_mean_raw);
    const half* k_mean = reinterpret_cast<const half*>(k_mean_raw);
    int32_t* topk_out = reinterpret_cast<int32_t*>(topk_out_raw);

    if (num_kv_blocks > 2048) {
        fprintf(stderr, "block_mean_topk: num_kv_blocks=%d > 2048 not supported\n", num_kv_blocks);
        return;
    }

    if (head_dim == 64) {
        if (causal) {
            dispatch_block_mean_topk<true, 64>(
                q_mean, k_mean, topk_out, flat_heads, num_q_blocks, num_kv_blocks, topk_count);
        } else {
            dispatch_block_mean_topk<false, 64>(
                q_mean, k_mean, topk_out, flat_heads, num_q_blocks, num_kv_blocks, topk_count);
        }
    } else if (head_dim == 128) {
        if (causal) {
            dispatch_block_mean_topk<true, 128>(
                q_mean, k_mean, topk_out, flat_heads, num_q_blocks, num_kv_blocks, topk_count);
        } else {
            dispatch_block_mean_topk<false, 128>(
                q_mean, k_mean, topk_out, flat_heads, num_q_blocks, num_kv_blocks, topk_count);
        }
    } else {
        fprintf(stderr, "block_mean_topk: unsupported head_dim=%d\n", head_dim);
    }
}

void single_query_key_mean_topk(
    const void* q_grouped_raw,
    const void* k_mean_raw,
    void* topk_out_raw,
    int flat_heads,
    int groups,
    int num_kv_blocks,
    int k_mean_capacity_blocks,
    int head_dim,
    int topk_count) {
    const half* q_grouped = reinterpret_cast<const half*>(q_grouped_raw);
    const half* k_mean = reinterpret_cast<const half*>(k_mean_raw);
    int32_t* topk_out = reinterpret_cast<int32_t*>(topk_out_raw);

    if (num_kv_blocks > 2048) {
        fprintf(stderr, "single_query_key_mean_topk: num_kv_blocks=%d > 2048 not supported\n",
                num_kv_blocks);
        return;
    }

    if (head_dim == 64) {
        dispatch_single_query_key_mean_topk<64>(
            q_grouped, k_mean, topk_out, flat_heads, groups, num_kv_blocks,
            k_mean_capacity_blocks, topk_count);
    } else if (head_dim == 128) {
        dispatch_single_query_key_mean_topk<128>(
            q_grouped, k_mean, topk_out, flat_heads, groups, num_kv_blocks,
            k_mean_capacity_blocks, topk_count);
    } else {
        fprintf(stderr, "single_query_key_mean_topk: unsupported head_dim=%d\n", head_dim);
    }
}

__global__ __launch_bounds__(256)
void pack_topk_mask_kernel(
    const int32_t* __restrict__ topk,
    unsigned long long* __restrict__ topk_mask,
    int topk_count,
    int total_units,
    int word_count) {
    const int row_id = blockIdx.x;
    const int tid = threadIdx.x;
    unsigned long long* mask_row = topk_mask + static_cast<int64_t>(row_id) * word_count;

    for (int word = tid; word < word_count; word += blockDim.x) {
        mask_row[word] = 0ULL;
    }
    __syncthreads();

    const int32_t* topk_row = topk + static_cast<int64_t>(row_id) * topk_count;
    for (int i = tid; i < topk_count; i += blockDim.x) {
        const int unit_id = topk_row[i];
        if (unit_id >= 0 && unit_id < total_units) {
            const int word = unit_id >> 6;
            if (word < word_count) {
                atomicOr(mask_row + word, 1ULL << (unit_id & 63));
            }
        }
    }
}

void pack_topk_mask(
    const int32_t* topk,
    void* topk_mask_raw,
    int row_count,
    int topk_count,
    int total_units,
    int word_count) {
    auto topk_mask = reinterpret_cast<unsigned long long*>(topk_mask_raw);
    if (row_count <= 0 || word_count <= 0) {
        return;
    }

    pack_topk_mask_kernel<<<row_count, 256>>>(
        topk, topk_mask, topk_count, total_units, word_count);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "pack_topk_mask kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}
