// The FP32 building blocks every host-side head in this module is written in terms of.
//
// They live here rather than inside `head.cpp` because the CLM-style student head computes with the
// same primitives, and this module's whole purpose is to be *the* number the device backend has to
// reproduce: two copies of `dot` or of `gelu` is two oracles. The bodies are unchanged from the
// original transcription, and `--selftest` re-checks the released head against the Python gold after
// the move so the extraction cannot shift a result silently.
//
// Deliberately small and dumb: the arithmetic is written in the open rather than hidden behind an
// expression template, because a reader has to be able to compare it line by line against PyTorch.

#pragma once

#include "joint_head/record.h"
#include "joint_head/weights.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::joint_head::detail {

struct Mat {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<float> data;

    [[nodiscard]] float* at(std::size_t row) noexcept { return data.data() + row * cols; }
    [[nodiscard]] const float* at(std::size_t row) const noexcept { return data.data() + row * cols; }
};

[[nodiscard]] inline Mat matrix(std::size_t rows, std::size_t cols) {
    Mat out;
    out.rows = rows;
    out.cols = cols;
    out.data.assign(rows * cols, 0.0F);
    return out;
}

[[nodiscard]] inline Mat rows_of(std::span<const float> values, std::size_t cols) {
    if (values.size() % cols != 0) {
        throw std::invalid_argument("joint head: " + std::to_string(values.size()) +
                                    " values are not a whole number of " + std::to_string(cols) +
                                    "-wide rows");
    }
    Mat out;
    out.rows = values.size() / cols;
    out.cols = cols;
    out.data.assign(values.begin(), values.end());
    return out;
}

// Eight partial sums. The wide reductions (5120 for the projections, 4096 for the residual MLP) are
// the entire cost of this reference, and one long dependency chain is both slow and a worse match for
// PyTorch's blocked accumulation than independent partials are.
[[nodiscard]] inline float dot(const float* left, const float* right, std::size_t count) noexcept {
    float s0 = 0.0F;
    float s1 = 0.0F;
    float s2 = 0.0F;
    float s3 = 0.0F;
    float s4 = 0.0F;
    float s5 = 0.0F;
    float s6 = 0.0F;
    float s7 = 0.0F;
    std::size_t index = 0;
    for (; index + 8 <= count; index += 8) {
        s0 += left[index] * right[index];
        s1 += left[index + 1] * right[index + 1];
        s2 += left[index + 2] * right[index + 2];
        s3 += left[index + 3] * right[index + 3];
        s4 += left[index + 4] * right[index + 4];
        s5 += left[index + 5] * right[index + 5];
        s6 += left[index + 6] * right[index + 6];
        s7 += left[index + 7] * right[index + 7];
    }
    for (; index < count; ++index) {
        s0 += left[index] * right[index];
    }
    return ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
}

// A row-major [out_features, in_features] block. Weight views let one `in_proj_weight` serve q, k and
// v without copying, exactly as F.linear slices it in the release module.
struct WeightView {
    const float* data = nullptr;
    std::size_t rows = 0;
    std::size_t cols = 0;

    [[nodiscard]] const float* row(std::size_t index) const noexcept { return data + index * cols; }
    [[nodiscard]] WeightView block(std::size_t first_row, std::size_t row_count) const noexcept {
        return WeightView {data + first_row * cols, row_count, cols};
    }
};

[[nodiscard]] inline WeightView weight_view(const Tensor& tensor, std::string_view name) {
    if (tensor.dims.size() != 2) {
        throw std::invalid_argument("joint head: '" + std::string(name) + "' is rank " +
                                    std::to_string(tensor.dims.size()) + ", a Linear needs rank 2");
    }
    return WeightView {tensor.values.data(), static_cast<std::size_t>(tensor.dims[0]),
                       static_cast<std::size_t>(tensor.dims[1])};
}

using BiasView = std::span<const float>;

[[nodiscard]] inline Mat linear(const Mat& input, WeightView weight, BiasView bias) {
    if (input.cols != weight.cols) {
        throw std::invalid_argument("joint head: Linear input is " + std::to_string(input.cols) +
                                    " wide, weights want " + std::to_string(weight.cols));
    }
    if (!bias.empty() && bias.size() != weight.rows) {
        throw std::invalid_argument("joint head: Linear bias has " + std::to_string(bias.size()) +
                                    " entries for " + std::to_string(weight.rows) + " outputs");
    }
    Mat output = matrix(input.rows, weight.rows);
    for (std::size_t row = 0; row < input.rows; ++row) {
        const float* source = input.at(row);
        float* target = output.at(row);
        for (std::size_t column = 0; column < weight.rows; ++column) {
            target[column] = dot(source, weight.row(column), weight.cols);
            if (!bias.empty()) {
                target[column] += bias[column];
            }
        }
    }
    return output;
}

[[nodiscard]] inline std::vector<float> linear_one(const float* source, WeightView weight,
                                                   BiasView bias) {
    std::vector<float> output(weight.rows, 0.0F);
    for (std::size_t column = 0; column < weight.rows; ++column) {
        output[column] = dot(source, weight.row(column), weight.cols);
        if (!bias.empty()) {
            output[column] += bias[column];
        }
    }
    return output;
}

// nn.LayerNorm: biased variance over the last axis, eps inside the reciprocal square root.
inline void layer_norm(const Mat& input, const float* weight, const float* bias, float epsilon,
                       Mat& output) {
    if (input.cols != output.cols || input.rows != output.rows) {
        throw std::logic_error("joint head: LayerNorm shape was not prepared");
    }
    const std::size_t width = input.cols;
    for (std::size_t row = 0; row < input.rows; ++row) {
        const float* source = input.at(row);
        float* target = output.at(row);
        double mean = 0.0;
        for (std::size_t column = 0; column < width; ++column) {
            mean += source[column];
        }
        mean /= static_cast<double>(width);
        double variance = 0.0;
        for (std::size_t column = 0; column < width; ++column) {
            const double centred = static_cast<double>(source[column]) - mean;
            variance += centred * centred;
        }
        variance /= static_cast<double>(width);
        const float scale = static_cast<float>(1.0 / std::sqrt(variance + epsilon));
        const float shift = static_cast<float>(-mean) * scale;
        for (std::size_t column = 0; column < width; ++column) {
            target[column] = source[column] * scale * weight[column] + bias[column] +
                             shift * weight[column];
        }
    }
}

[[nodiscard]] inline Mat layer_norm(const Mat& input, const Tensor& weight, const Tensor& bias,
                                    float epsilon) {
    Mat output = matrix(input.rows, input.cols);
    layer_norm(input, weight.values.data(), bias.values.data(), epsilon, output);
    return output;
}

inline void softmax_inplace(float* values, std::size_t count) {
    float peak = values[0];
    for (std::size_t index = 1; index < count; ++index) {
        peak = std::max(peak, values[index]);
    }
    double total = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        values[index] = static_cast<float>(std::exp(static_cast<double>(values[index]) -
                                                    static_cast<double>(peak)));
        total += values[index];
    }
    const float inverse = static_cast<float>(1.0 / total);
    for (std::size_t index = 0; index < count; ++index) {
        values[index] *= inverse;
    }
}

// nn.GELU's default (approximate='none'): the exact erf form.
[[nodiscard]] inline float gelu(float value) noexcept {
    return 0.5F * value * (1.0F + std::erf(value * 0.70710678118654752440F));
}

inline void add_rows(const Mat& addend, Mat& target) {
    if (addend.rows != target.rows || addend.cols != target.cols) {
        throw std::logic_error("joint head: residual add is " + std::to_string(addend.rows) + "x" +
                               std::to_string(addend.cols) + " onto a " +
                               std::to_string(target.rows) + "x" + std::to_string(target.cols) +
                               " stage");
    }
    for (std::size_t row = 0; row < target.rows; ++row) {
        float* destination = target.at(row);
        const float* source = addend.at(row);
        for (std::size_t column = 0; column < target.cols; ++column) {
            destination[column] += source[column];
        }
    }
}

// `_mean_span` in the release module: mean over hidden rows, with a degenerate span widened to one
// row the same way Python's `if end <= start` does.
[[nodiscard]] inline Mat mean_rows(const Mat& source, const std::vector<Span>& spans,
                                   std::string_view what) {
    Mat output = matrix(spans.size(), source.cols);
    for (std::size_t index = 0; index < spans.size(); ++index) {
        const Span span = spans[index];
        if (span.end <= span.begin || span.end > source.rows) {
            throw std::invalid_argument("joint head: " + std::string(what) + " span [" +
                                        std::to_string(span.begin) + "," +
                                        std::to_string(span.end) + ") is outside " +
                                        std::to_string(source.rows) + " rows");
        }
        float* target = output.at(index);
        for (std::size_t column = 0; column < source.cols; ++column) {
            target[column] = 0.0F;
        }
        for (std::size_t row = span.begin; row < span.end; ++row) {
            const float* source_row = source.at(row);
            for (std::size_t column = 0; column < source.cols; ++column) {
                target[column] += source_row[column];
            }
        }
        const float inverse = 1.0F / static_cast<float>(span.length());
        for (std::size_t column = 0; column < source.cols; ++column) {
            target[column] *= inverse;
        }
    }
    return output;
}

// `output_embedding_weight[token_ids].mean(dim=0)` for one option's token range.
[[nodiscard]] inline std::vector<float> mean_lexical(LexicalTable& lexical,
                                                     const std::vector<std::int32_t>& token_ids,
                                                     Span span, std::size_t width,
                                                     std::string_view what) {
    if (span.end <= span.begin || span.end > token_ids.size()) {
        throw std::invalid_argument("joint head: " + std::string(what) + " token span [" +
                                    std::to_string(span.begin) + "," + std::to_string(span.end) +
                                    ") is outside " + std::to_string(token_ids.size()) + " tokens");
    }
    if (static_cast<std::size_t>(lexical.columns()) != width) {
        throw std::invalid_argument("joint head: lexical table is " +
                                    std::to_string(lexical.columns()) + " wide, hidden is " +
                                    std::to_string(width));
    }
    std::vector<float> target(width, 0.0F);
    for (std::size_t index = span.begin; index < span.end; ++index) {
        const std::span<const float> row = lexical.row(token_ids[index]);
        for (std::size_t column = 0; column < width; ++column) {
            target[column] += row[column];
        }
    }
    const float inverse = 1.0F / static_cast<float>(span.length());
    for (float& value : target) {
        value *= inverse;
    }
    return target;
}

// functional.normalize / cosine_similarity both floor the norm at eps=1e-12.
[[nodiscard]] inline std::vector<float> l2_normalize(std::span<const float> values) {
    double squared = 0.0;
    for (const float value : values) {
        squared += static_cast<double>(value) * static_cast<double>(value);
    }
    const float norm = static_cast<float>(std::sqrt(squared));
    const float divisor = std::max(norm, 1e-12F);
    std::vector<float> out(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        out[index] = values[index] / divisor;
    }
    return out;
}

} // namespace ninfer::joint_head::detail
