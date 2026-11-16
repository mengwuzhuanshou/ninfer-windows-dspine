#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Destination for the line-D page representative K vector.
 *
 * page_sums is device FP32 [pages, kv_heads, head_dim] and page_counts device I32 [pages]; the
 * caller zeroes both for the written range before every call. A sum/count pair rather than a mean
 * is deliberate: a logical page can be split across prefill chunks, and adding the two partial
 * contributions on the host gives the same mean as one whole-page accumulation.
 */
struct KvmemPageIndexTarget {
    float* page_sums        = nullptr;
    std::int32_t* page_counts = nullptr;
    std::int32_t page_base  = 0;
    std::int32_t pages      = 0;
    std::int32_t page_shift = 6;
};

/**
 * KVMem line D (scoring shape ported from kvmem/qvmem-qw3 block_kmean_paged_kernel, Apache-2.0):
 * accumulate every K column of one prefill chunk into its logical page's representative vector.
 *
 * k is BF16 [head_dim, kv_heads, T] with byte strides nb[0..2]; positions is device I32 [T] holding
 * each column's logical token position in its sequence. Column t contributes to slot
 * (positions[t] >> page_shift) - page_base; columns outside [0,pages) are ignored, which is what
 * lets one fixed scratch serve a chunk that straddles a page boundary. The kernel is purely
 * additive (it never reads a destination slot before writing it), so repeated calls over the same
 * target accumulate. It mutates no input tensor and the model's numerics are untouched: this reads
 * the pre-RoPE K the attention op would consume next.
 */
void kvmem_page_index_accumulate(const Tensor& k, const Tensor& positions,
                                 const KvmemPageIndexTarget& target, cudaStream_t stream);

} // namespace ninfer::ops
