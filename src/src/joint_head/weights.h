// Weight framing for the joint schema head: everything the C++ head needs that does not come from a
// .ninfer artifact.
//
// Two sidecars, both written by `tools/export_joint_head.py` from the Clef release:
//
//   HeadWeights   the 128M-parameter joint schema head, copied verbatim in BF16 and widened to FP32
//                 once at open(). The head is not a backbone tower: it has no artifact representation,
//                 so this is its product load path, not a test convenience.
//   LexicalTable  the raw BF16 `lm_head` matrix. The head indexes it by absolute vocabulary id for
//                 the option-lexical term, and the artifact stores that matrix after an FP8 round
//                 trip, so the release table is the exact source. Rows are fixed size and seekable;
//                 rows are widened and cached on demand because a record touches a few hundred of the
//                 248k.

#pragma once

#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ninfer::joint_head {

struct Tensor {
    std::vector<std::int64_t> dims;
    std::vector<float> values;

    [[nodiscard]] std::size_t numel() const noexcept { return values.size(); }
    [[nodiscard]] std::string shape_text() const;
};

class HeadWeights {
public:
    [[nodiscard]] static HeadWeights open(const std::string& path);

    HeadWeights(HeadWeights&&) noexcept = default;
    HeadWeights& operator=(HeadWeights&&) noexcept = default;
    HeadWeights(const HeadWeights&) = delete;
    HeadWeights& operator=(const HeadWeights&) = delete;

    // Throws with the tensor name and the file it came from: a missing projection is a wrong sidecar,
    // and the forward would otherwise fail as a NaN several stages later.
    [[nodiscard]] const Tensor& tensor(std::string_view name) const;
    [[nodiscard]] float scalar(std::string_view name) const;
    [[nodiscard]] bool contains(std::string_view name) const noexcept;
    [[nodiscard]] std::size_t parameter_count() const noexcept;
    [[nodiscard]] const std::string& path() const noexcept { return path_; }

private:
    // Declared because the move constructor below suppresses the implicit one: open() builds the
    // image locally and moves it out, and there is no other way to be holding a head weight set.
    HeadWeights() = default;

    std::string path_;
    std::unordered_map<std::string, Tensor> tensors_;
};

// Validate the shapes the forward actually multiplies against, before any of them run. The sidecar
// carries its own dims, so a mismatch means the file is not this head (or not this version of it).
void validate_head_shapes(const HeadWeights& weights, int hidden_size, int width, int heads,
                          int feedforward, int routing_layers, int layers);

class LexicalTable {
public:
    [[nodiscard]] static LexicalTable open(const std::string& path);

    LexicalTable(LexicalTable&&) noexcept;
    LexicalTable& operator=(LexicalTable&&) noexcept;
    LexicalTable(const LexicalTable&) = delete;
    LexicalTable& operator=(const LexicalTable&) = delete;
    ~LexicalTable();

    // Row `id` of the released BF16 table as FP32. Cached: the head asks for the same option tokens
    // across questions and records.
    [[nodiscard]] std::span<const float> row(std::int64_t id);

    [[nodiscard]] std::int64_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::int64_t columns() const noexcept { return columns_; }
    [[nodiscard]] std::size_t cached_rows() const noexcept { return cache_.size(); }
    [[nodiscard]] const std::string& path() const noexcept { return path_; }

private:
    // open() owns the only valid construction: a table is a checked header plus an open handle, so
    // there is no default-constructed half-state for the rest of the head to trip over.
    LexicalTable(std::string path, std::FILE* handle, std::int64_t rows, std::int64_t columns,
                 std::uint64_t data_offset) noexcept;

    std::string path_;
    std::FILE* handle_ = nullptr;
    std::int64_t rows_ = 0;
    std::int64_t columns_ = 0;
    std::uint64_t data_offset_ = 0;
    std::unordered_map<std::int64_t, std::vector<float>> cache_;
};

} // namespace ninfer::joint_head
