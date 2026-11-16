#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

// One block per (token column, kv head), one thread per head dimension. Every element is read once
// and lands in exactly one page slot, so the whole chunk costs T*kv_heads*head_dim atomic adds
// rather than a per-page rescan of the chunk.
__global__ void kvmem_page_index_accumulate_kernel(const char* __restrict__ k,
                                                   const std::int32_t* __restrict__ positions,
                                                   float* __restrict__ page_sums,
                                                   std::int32_t* __restrict__ page_counts,
                                                   std::int64_t stride_d, std::int64_t stride_h,
                                                   std::int64_t stride_t, int head_dim, int kv_heads,
                                                   int page_base, int pages, int page_shift) {
    const int token = blockIdx.x;
    const int head  = blockIdx.y;
    if (head >= kv_heads) { return; }
    const int dim = threadIdx.x;
    if (dim >= head_dim) { return; }
    const int page = positions[token] >> page_shift;
    const int slot = page - page_base;
    if (slot < 0 || slot >= pages) { return; }
    const float value = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
        k + static_cast<std::int64_t>(token) * stride_t + static_cast<std::int64_t>(head) * stride_h +
        static_cast<std::int64_t>(dim) * stride_d));
    atomicAdd(&page_sums[(static_cast<std::int64_t>(slot) * kv_heads + head) * head_dim + dim], value);
    if (dim == 0) { atomicAdd(&page_counts[slot], 1); }
}

} // namespace ninfer::ops
