#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cstdint>

namespace ninfer::ops::detail {

struct Nvfp4WeightGeometry {
    std::uint64_t code_plane_bytes;
    std::uint64_t scale_plane_offset;
    std::uint64_t scale_plane_bytes;
    std::uint64_t required_payload_bytes;
};

Nvfp4WeightGeometry validate_nvfp4_weight(const Weight& weight, const char* operation);

// Validates a 128-row-aligned row slice of a packed NVFP4 parent (the DSpine/DFlash2 context K/V
// views). The code and scale planes keep the registered arrangement but sit off the canonical
// single-payload offsets, so the plane equality check is replaced by an ordering check.
Nvfp4WeightGeometry validate_nvfp4_view_weight(const Weight& weight, const char* operation);

} // namespace ninfer::ops::detail
