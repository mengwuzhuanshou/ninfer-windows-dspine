#pragma once
#include "core/device.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/tiled_mma.cuh"
#include <stdexcept>

namespace ninfer::ops::detail {
template <class G, class S>
void launch_nvfp4_kv_tiled_mma(const CausalAttentionOperands& p, Nvfp4KvReadView cache,
                               cudaStream_t stream) {
    validate_quantized_causal_operands<G>(p, cache);
    if (p.batch != 1)
        throw std::invalid_argument("NVFP4 tiled attention requires a complete single query row");
    const auto invoke = [&]<class Metadata>(Metadata metadata) {
        constexpr auto kernel    = nvfp4_kv_tiled_mma_kernel<G, S, Metadata>;
        static const auto status = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, S::kSharedBytes);
        CUDA_CHECK(status);
        const dim3 grid(div_up(p.width, S::kQueryRows), G::QHeads);
        kernel<<<grid, S::kThreads, S::kSharedBytes, stream>>>(
            p.q, cache.keys, cache.values, cache.key_scales, cache.value_scales, metadata,
            p.positions, p.scale, p.out, p.width);
        CUDA_CHECK(cudaGetLastError());
    };
    // KVMem route 3 (ported from shensanshu/ninfer-master-shensanshu-kvmem, Apache-2.0): the block
    // retrieval mask rides on the metadata struct's defaulted last member.
    if (!cache.table_rows)
        invoke(PagedKVDirectMetadata{cache.tables, cache.selected_blocks});
    else if (cache.valid_columns)
        invoke(PagedKVBatchMetadata<true>{cache.tables, cache.valid_columns, cache.table_rows,
                                          cache.table_stride, cache.selected_blocks});
    else
        invoke(PagedKVBatchMetadata<false>{cache.tables, nullptr, cache.table_rows,
                                           cache.table_stride, cache.selected_blocks});
}

} // namespace ninfer::ops::detail
