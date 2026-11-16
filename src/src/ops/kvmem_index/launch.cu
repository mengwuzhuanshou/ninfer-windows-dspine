// ninfer::ops - KVMem line D page-index harvest.
#include "ninfer/ops/kvmem_index.h"
#include "ops/kvmem_index/kernel.cuh"

#include "core/device.h"
#include "core/dtype.h"

#include <stdexcept>

namespace ninfer::ops {

void kvmem_page_index_accumulate(const Tensor& k, const Tensor& positions,
                                 const KvmemPageIndexTarget& target, cudaStream_t stream) {
    const int head_dim = k.ne[0];
    const int kv_heads = k.ne[1];
    const int tokens   = k.ne[2];
    if (k.dtype != DType::BF16 || positions.dtype != DType::I32) {
        throw std::invalid_argument("kvmem page index needs BF16 K and I32 positions");
    }
    if (head_dim <= 0 || kv_heads <= 0 || tokens <= 0) {
        throw std::invalid_argument("kvmem page index got an empty K tensor");
    }
    if (head_dim > 1024) {
        throw std::invalid_argument("kvmem page index head dimension exceeds one block of threads");
    }
    if (positions.ne[0] < tokens) {
        throw std::invalid_argument("kvmem page index position count is below the K column count");
    }
    if (target.page_sums == nullptr || target.page_counts == nullptr || target.pages <= 0 ||
        target.page_shift < 0) {
        throw std::invalid_argument("kvmem page index target is not a written range");
    }
    if (tokens > 65535) {
        throw std::invalid_argument("kvmem page index chunk exceeds the grid.x token bound");
    }
    const dim3 blocks(tokens, kv_heads);
    const dim3 threads(head_dim);
    kvmem_page_index_accumulate_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const char*>(k.data), static_cast<const std::int32_t*>(positions.data),
        target.page_sums, target.page_counts, k.nb[0], k.nb[1], k.nb[2], head_dim, kv_heads,
        target.page_base, target.pages, target.page_shift);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
