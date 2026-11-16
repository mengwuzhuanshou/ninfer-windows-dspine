#include "joint_head/weights.h"

#include "joint_head/bf16.h"
#include "joint_head/student.h"  // StudentConfig, for the shape table the student forward needs

#include <array>
#include <cstddef>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::joint_head {
namespace {

constexpr std::array<char, 8> kHeadMagic {'N', 'I', 'N', 'H', 'H', 'E', 'A', 'D'};
constexpr std::array<char, 8> kLexicalMagic {'N', 'I', 'N', 'H', 'L', 'E', 'X', '\0'};
constexpr std::uint32_t kSidecarVersion = 1;
constexpr std::uint32_t kTypeBFloat16 = 0;
constexpr std::uint32_t kTypeFloat32 = 1;

// Little-endian scalars straight out of the image; the framer writes "<" formats and the only
// supported platform is x86-64, but read through memcpy so the code never makes an alignment claim.
template <typename T>
[[nodiscard]] T read_scalar(const std::uint8_t* source) noexcept {
    T value {};
    std::memcpy(&value, source, sizeof(T));
    return value;
}

// Seek wider than `long`. MSVC's long is 32-bit, so std::fseek cannot address past 2 GiB -- and the
// released BF16 lm_head table is 2.5 GB, which is exactly the range this file has to read row-wise.
// POSIX gets this for free because its long is 64-bit.
[[nodiscard]] int seek64(std::FILE* handle, std::int64_t offset, int origin) {
#if defined(_WIN32)
    return ::_fseeki64(handle, offset, origin);
#else
    return std::fseek(handle, static_cast<long>(offset), origin);
#endif
}

[[nodiscard]] std::int64_t tell64(std::FILE* handle) {
#if defined(_WIN32)
    return ::_ftelli64(handle);
#else
    return static_cast<std::int64_t>(std::ftell(handle));
#endif
}

[[nodiscard]] std::vector<std::uint8_t> slurp(const std::string& path) {
    std::FILE* handle = std::fopen(path.c_str(), "rb");
    if (handle == nullptr) {
        throw std::invalid_argument("joint head: cannot open " + path);
    }
    std::vector<std::uint8_t> image;
    if (seek64(handle, 0, SEEK_END) != 0) {
        std::fclose(handle);
        throw std::invalid_argument("joint head: cannot size " + path);
    }
    const std::int64_t size = tell64(handle);
    std::rewind(handle);
    if (size <= 0) {
        std::fclose(handle);
        throw std::invalid_argument("joint head: " + path + " is empty");
    }
    image.resize(static_cast<std::size_t>(size));
    const std::size_t read = std::fread(image.data(), 1, image.size(), handle);
    std::fclose(handle);
    if (read != image.size()) {
        throw std::invalid_argument("joint head: short read on " + path + " (" +
                                    std::to_string(read) + " of " + std::to_string(image.size()) +
                                    " bytes); rewrite it with tools/export_joint_head.py");
    }
    return image;
}

[[nodiscard]] std::string to_string(std::string_view view) {
    return std::string(view);
}

// One expected tensor. Shared by both head validators: a sidecar that is missing a projection or
// carries it at the wrong shape is a different head, and the failure has to name the file and the
// tensor rather than surface as a NaN three stages later.
void require_shape(const HeadWeights& weights, const std::string& name,
                   const std::vector<std::int64_t>& dims) {
    if (!weights.contains(name)) {
        throw std::invalid_argument("joint head: sidecar " + weights.path() + " is missing '" +
                                    name + "'");
    }
    const Tensor& tensor = weights.tensor(name);
    if (tensor.dims != dims) {
        throw std::invalid_argument("joint head: '" + name + "' is " + tensor.shape_text() + " in " +
                                    weights.path() + ", the head config needs" +
                                    [&dims] {
                                        std::ostringstream text;
                                        text << " [";
                                        for (std::size_t axis = 0; axis < dims.size(); ++axis) {
                                            if (axis != 0) {
                                                text << ',';
                                            }
                                            text << dims[axis];
                                        }
                                        text << "]";
                                        return text.str();
                                    }());
    }
}

} // namespace

std::string Tensor::shape_text() const {
    std::ostringstream text;
    text << '[';
    for (std::size_t axis = 0; axis < dims.size(); ++axis) {
        if (axis != 0) {
            text << ',';
        }
        text << dims[axis];
    }
    text << ']';
    return text.str();
}

HeadWeights HeadWeights::open(const std::string& path) {
    const std::vector<std::uint8_t> image = slurp(path);
    const std::uint8_t* base = image.data();
    std::size_t cursor = 0;

    auto take = [&](std::size_t count) {
        if (cursor + count > image.size()) {
            throw std::invalid_argument("joint head: " + path + " ends inside its header");
        }
        const std::uint8_t* at = base + cursor;
        cursor += count;
        return at;
    };

    if (std::memcmp(take(8), kHeadMagic.data(), 8) != 0) {
        throw std::invalid_argument("joint head: " + path + " is not a NINHHEAD sidecar");
    }
    const std::uint32_t version = read_scalar<std::uint32_t>(take(4));
    if (version != kSidecarVersion) {
        throw std::invalid_argument("joint head: unsupported sidecar version " +
                                    std::to_string(version) + " in " + path);
    }
    const std::uint32_t count = read_scalar<std::uint32_t>(take(4));
    (void)take(4); // reserved

    HeadWeights weights;
    weights.path_ = path;
    for (std::uint32_t entry = 0; entry < count; ++entry) {
        const std::uint32_t name_length = read_scalar<std::uint32_t>(take(4));
        const std::string_view name(reinterpret_cast<const char*>(take(name_length)), name_length);
        const std::uint32_t rank = read_scalar<std::uint32_t>(take(4));
        Tensor tensor;
        tensor.dims.reserve(rank);
        for (std::uint32_t axis = 0; axis < rank; ++axis) {
            tensor.dims.push_back(read_scalar<std::int64_t>(take(8)));
        }
        const std::uint32_t type = read_scalar<std::uint32_t>(take(4));
        const std::uint64_t offset = read_scalar<std::uint64_t>(take(8));
        const std::uint64_t nbytes = read_scalar<std::uint64_t>(take(8));
        if (offset + nbytes > image.size()) {
            throw std::invalid_argument("joint head: tensor " + to_string(name) +
                                        " runs past the end of " + path);
        }
        const std::size_t elements = nbytes / (type == kTypeBFloat16 ? 2U : 4U);
        tensor.values.resize(elements);
        if (type == kTypeBFloat16) {
            for (std::size_t index = 0; index < elements; ++index) {
                tensor.values[index] =
                    to_float(read_scalar<std::uint16_t>(base + offset + index * 2));
            }
        } else if (type == kTypeFloat32) {
            std::memcpy(tensor.values.data(), base + offset, nbytes);
        } else {
            throw std::invalid_argument("joint head: tensor " + to_string(name) + " has type " +
                                        std::to_string(type) + "; the sidecar only writes 0=BF16");
        }
        if (!weights.tensors_.emplace(to_string(name), std::move(tensor)).second) {
            throw std::invalid_argument("joint head: duplicate tensor " + to_string(name) + " in " +
                                        path);
        }
    }
    if (weights.tensors_.empty()) {
        throw std::invalid_argument("joint head: " + path + " carries no tensors");
    }
    return weights;
}

bool HeadWeights::contains(std::string_view name) const noexcept {
    return tensors_.find(to_string(name)) != tensors_.end();
}

const Tensor& HeadWeights::tensor(std::string_view name) const {
    const auto found = tensors_.find(to_string(name));
    if (found == tensors_.end()) {
        throw std::invalid_argument("joint head: sidecar " + path_ + " has no tensor '" +
                                    to_string(name) + "'");
    }
    return found->second;
}

float HeadWeights::scalar(std::string_view name) const {
    const Tensor& entry = tensor(name);
    if (entry.numel() != 1) {
        throw std::invalid_argument("joint head: '" + to_string(name) + "' holds " +
                                    std::to_string(entry.numel()) + " values, expected a scalar");
    }
    return entry.values.front();
}

std::size_t HeadWeights::parameter_count() const noexcept {
    std::size_t total = 0;
    for (const auto& [name, tensor] : tensors_) {
        total += tensor.numel();
    }
    return total;
}

void validate_head_shapes(const HeadWeights& weights, int hidden_size, int width, int heads,
                          int feedforward, int routing_layers, int layers) {
    if (width % heads != 0) {
        throw std::invalid_argument("joint head: width " + std::to_string(width) +
                                    " is not a multiple of " + std::to_string(heads) + " heads");
    }
    const auto h = static_cast<std::int64_t>(hidden_size);
    const auto w = static_cast<std::int64_t>(width);
    const auto f = static_cast<std::int64_t>(feedforward);

    std::vector<std::pair<std::string, std::vector<std::int64_t>>> expected;
    const auto push = [&expected](std::string name, std::vector<std::int64_t> dims) {
        expected.emplace_back(std::move(name), std::move(dims));
    };
    push("hidden_norm.weight", {h});
    push("hidden_norm.bias", {h});
    for (const char* projection :
         {"memory_projection", "question_projection", "option_question_projection",
          "global_projection", "option_context_projection", "option_lexical_projection"}) {
        push(std::string(projection) + ".weight", {w, h});
    }
    push("type_embedding.weight", {3, w});
    for (int layer = 0; layer < routing_layers; ++layer) {
        const std::string prefix = "evidence_layers." + std::to_string(layer) + ".";
        for (const char* norm : {"query_norm", "memory_norm", "feedforward_norm"}) {
            push(prefix + norm + ".weight", {w});
            push(prefix + norm + ".bias", {w});
        }
        push(prefix + "attention.in_proj_weight", {3 * w, w});
        push(prefix + "attention.in_proj_bias", {3 * w});
        push(prefix + "attention.out_proj.weight", {w, w});
        push(prefix + "attention.out_proj.bias", {w});
        push(prefix + "feedforward.0.weight", {f, w});
        push(prefix + "feedforward.0.bias", {f});
        push(prefix + "feedforward.3.weight", {w, f});
        push(prefix + "feedforward.3.bias", {w});
    }
    for (const char* norm : {"option_summary_norm", "field_norm", "option_norm"}) {
        push(std::string(norm) + ".weight", {w});
        push(std::string(norm) + ".bias", {w});
    }
    for (int layer = 0; layer < layers; ++layer) {
        const std::string prefix = "layers." + std::to_string(layer) + ".";
        for (const char* norm : {"norm1", "norm2", "norm3"}) {
            push(prefix + norm + ".weight", {w});
            push(prefix + norm + ".bias", {w});
        }
        for (const char* attention : {"self_attn", "multihead_attn"}) {
            push(prefix + attention + ".in_proj_weight", {3 * w, w});
            push(prefix + attention + ".in_proj_bias", {3 * w});
            push(prefix + attention + ".out_proj.weight", {w, w});
            push(prefix + attention + ".out_proj.bias", {w});
        }
        push(prefix + "linear1.weight", {f, w});
        push(prefix + "linear1.bias", {f});
        push(prefix + "linear2.weight", {w, f});
        push(prefix + "linear2.bias", {w});
    }
    push("residual_scorer.0.weight", {w, 4 * w});
    push("residual_scorer.0.bias", {w});
    push("residual_scorer.3.weight", {1, w});
    push("residual_scorer.3.bias", {1});
    for (const char* parameter : {"prior_logit_scale", "joint_logit_scale", "residual_gate"}) {
        push(parameter, {});
    }

    for (const auto& [name, dims] : expected) {
        require_shape(weights, name, dims);
    }
}

LexicalTable LexicalTable::open(const std::string& path) {
    std::FILE* handle = std::fopen(path.c_str(), "rb");
    if (handle == nullptr) {
        throw std::invalid_argument("lexical table: cannot open " + path);
    }
    auto fail = [&path](const std::string& reason) {
        return std::invalid_argument("lexical table: " + path + " " + reason);
    };
    std::array<char, 8> magic {};
    if (std::fread(magic.data(), 1, magic.size(), handle) != magic.size() ||
        magic != kLexicalMagic) {
        std::fclose(handle);
        throw fail("is not a NINHLEX table");
    }
    std::uint32_t version = 0;
    std::uint32_t rows = 0;
    std::uint32_t columns = 0;
    std::uint32_t type = 0;
    std::uint32_t reserved = 0;
    std::uint64_t data_offset = 0;
    std::uint64_t data_bytes = 0;
    // One read per statement. Folding these into a single sum expression was undefined behaviour: the
    // calls are unsequenced relative to each other and all advance the same file position, so the
    // compiler is free to hand the fields out in an arbitrary order -- which is what it did, producing
    // "unsupported version 0" for a header whose version field is 1.
    std::size_t header = 0;
    header += std::fread(&version, 4, 1, handle) * 4;
    header += std::fread(&rows, 4, 1, handle) * 4;
    header += std::fread(&columns, 4, 1, handle) * 4;
    header += std::fread(&type, 4, 1, handle) * 4;
    header += std::fread(&reserved, 4, 1, handle) * 4;
    header += std::fread(&data_offset, 8, 1, handle) * 8;
    header += std::fread(&data_bytes, 8, 1, handle) * 8;
    (void)reserved;
    if (header != 36) {
        std::fclose(handle);
        throw fail("ends inside its header");
    }
    if (version != kSidecarVersion) {
        std::fclose(handle);
        throw fail("has unsupported version " + std::to_string(version));
    }
    if (type != kTypeBFloat16) {
        std::fclose(handle);
        throw fail("stores type " + std::to_string(type) + "; only BF16 rows are supported");
    }
    if (rows == 0U || columns == 0U ||
        data_bytes != static_cast<std::uint64_t>(rows) * columns * 2U) {
        std::fclose(handle);
        throw fail("declares " + std::to_string(rows) + "x" + std::to_string(columns) +
                   " over " + std::to_string(data_bytes) + " bytes");
    }
    if (seek64(handle, static_cast<std::int64_t>(data_offset + data_bytes - 1), SEEK_SET) != 0 ||
        std::fgetc(handle) == EOF) {
        std::fclose(handle);
        throw fail("is shorter than its declared table");
    }
    return LexicalTable {path, handle, static_cast<std::int64_t>(rows),
                         static_cast<std::int64_t>(columns), data_offset};
}

LexicalTable::LexicalTable(std::string path, std::FILE* handle, std::int64_t rows,
                           std::int64_t columns, std::uint64_t data_offset) noexcept
    : path_(std::move(path)),
      handle_(handle),
      rows_(rows),
      columns_(columns),
      data_offset_(data_offset) {}

LexicalTable::LexicalTable(LexicalTable&& other) noexcept
    : path_(std::move(other.path_)),
      handle_(std::exchange(other.handle_, nullptr)),
      rows_(other.rows_),
      columns_(other.columns_),
      data_offset_(other.data_offset_),
      cache_(std::move(other.cache_)) {}

LexicalTable& LexicalTable::operator=(LexicalTable&& other) noexcept {
    if (this != &other) {
        if (handle_ != nullptr) {
            std::fclose(handle_);
        }
        path_ = std::move(other.path_);
        handle_ = std::exchange(other.handle_, nullptr);
        rows_ = other.rows_;
        columns_ = other.columns_;
        data_offset_ = other.data_offset_;
        cache_ = std::move(other.cache_);
    }
    return *this;
}

LexicalTable::~LexicalTable() {
    if (handle_ != nullptr) {
        std::fclose(handle_);
    }
}

std::span<const float> LexicalTable::row(std::int64_t id) {
    const auto cached = cache_.find(id);
    if (cached != cache_.end()) {
        return cached->second;
    }
    if (id < 0 || id >= rows_) {
        throw std::out_of_range("lexical table: vocabulary id " + std::to_string(id) +
                                " is outside the 0.." + std::to_string(rows_ - 1) + " table in " +
                                path_);
    }
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(columns_));
    const std::int64_t position = static_cast<std::int64_t>(
        data_offset_ + static_cast<std::uint64_t>(id) * static_cast<std::uint64_t>(columns_) * 2U);
    if (seek64(handle_, position, SEEK_SET) != 0 ||
        std::fread(bits.data(), 2, bits.size(), handle_) != bits.size()) {
        throw std::invalid_argument("lexical table: cannot read row " + std::to_string(id) +
                                    " of " + path_);
    }
    const auto [iterator, inserted] =
        cache_.emplace(static_cast<std::int64_t>(id), widen(std::span<const std::uint16_t>(bits)));
    (void)inserted;
    return iterator->second;
}

// The student sidecar is the same NINHHEAD container written by `train_clm_head.py`, with PyTorch's
// own module names inside it -- `state.0` / `state.2` are the two Linear layers of the state MLP and
// `option.1` is the one after the option GELU. Those names come from `nn.Sequential` indexing, so they
// are part of the file format: renaming a layer in Python renames a tensor here.
void validate_student_shapes(const HeadWeights& weights, const StudentConfig& config) {
    const auto h = static_cast<std::int64_t>(config.hidden_size);
    const auto w = static_cast<std::int64_t>(config.width);

    std::vector<std::pair<std::string, std::vector<std::int64_t>>> expected;
    const auto push = [&expected](std::string name, std::vector<std::int64_t> dims) {
        expected.emplace_back(std::move(name), std::move(dims));
    };
    push("state.0.weight", {w, 2 * h});
    push("state.0.bias", {w});
    push("state.2.weight", {w, w});
    push("state.2.bias", {w});
    push("type_embedding.weight", {static_cast<std::int64_t>(config.types), w});
    push("option_context.weight", {w, h});
    push("option_context.bias", {w});
    push("option_lexical.weight", {w, h});
    push("option_lexical.bias", {w});
    push("option.1.weight", {w, w});
    push("option.1.bias", {w});
    push("log_logit_scale", {});

    for (const auto& [name, dims] : expected) {
        require_shape(weights, name, dims);
    }
}

} // namespace ninfer::joint_head
