// BFloat16 widening for the host-side head reference.
//
// Only the lossless direction lives here: every BF16 value is exactly representable in FP32, so the
// reference head can hold released BF16 weights and engine-exported hidden states as FP32 without
// changing a number. The narrowing direction needs a rounding rule, and nothing in the CPU path
// narrows.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

namespace ninfer::joint_head {

[[nodiscard]] inline float to_float(std::uint16_t bits) noexcept {
    // A BF16 value is the top half of an FP32 word (sign, 8 exponent bits, 7 mantissa bits), so
    // widening is a shift. The result is a bit pattern, not an integer, so it travels through
    // memory rather than through a cast.
    const std::uint32_t widened = static_cast<std::uint32_t>(bits) << 16U;
    float value = 0.0F;
    std::memcpy(&value, &widened, sizeof(value));
    return value;
}

[[nodiscard]] inline std::vector<float> widen(std::span<const std::uint16_t> bits) {
    std::vector<float> out(bits.size());
    for (std::size_t index = 0; index < bits.size(); ++index) {
        out[index] = to_float(bits[index]);
    }
    return out;
}

} // namespace ninfer::joint_head
