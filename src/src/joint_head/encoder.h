// The record encoder: the port of `encode_record` from the release's `joint_schema_model.py`.
//
// It exists because the head is fed spans, not text. The schema block is tokenized fragment by fragment
// -- every `\nFIELD n\n`, every `OPTION n: `, the state, the chat prefix and suffix -- and each fragment
// starts a fresh BPE run, so the resulting ids are NOT what one encode of the concatenated prompt would
// give. That property is the whole interface: the encoder owes the head a token sequence plus the
// question and option spans inside it, and nothing else.
//
// Two Python behaviours carry over and are easy to lose:
//   * `record["questions"]` iterates in request order, so the JSON has to stay ordered. The option list
//     of a `score` question is enumerated by position, so re-sorting the keys silently relabels answers.
//   * `render()` is `json.dumps(..., sort_keys=True)` for anything that is not a string, so nested
//     objects canonicalise while the outer question order does not.
//
// The head module owns no tokenizer: the engine's BPE frontend is the only implementation, and it is
// injected through `FragmentEncoder` so the head library keeps linking nothing from the CUDA side.

#pragma once

#include "joint_head/record.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::joint_head {

using Request = nlohmann::ordered_json;

class FragmentEncoder {
public:
    FragmentEncoder() = default;
    FragmentEncoder(const FragmentEncoder&) = delete;
    FragmentEncoder& operator=(const FragmentEncoder&) = delete;
    virtual ~FragmentEncoder() = default;

    // One fragment, one BPE run: the equivalent of `tokenizer(text, add_special_tokens=False)`.
    [[nodiscard]] virtual std::vector<std::int32_t> encode(std::string_view fragment) const = 0;
};

struct EncodeRecordOptions {
    // The release default.
    std::size_t max_length = 16384;
    // Truncates the state before the budget check, which is how a caller keeps a cap on prompt cost.
    std::optional<std::size_t> max_state_tokens;
};

// `render` and `question_options` are exposed because the serving path and the parity harness both need
// them, and because a disagreement about them is a disagreement about the input contract.
[[nodiscard]] std::string render(const Request& value);

// The option list in the order the head will see it: (true, false) for noul, key-sorted for choice,
// positional for score. The description is a copy because a missing one has to arrive as null, and a
// pair of references would make the caller guess at lifetimes. `question_id` only appears in errors.
[[nodiscard]] std::vector<std::pair<std::string, Request>> question_options(
    const Request& question, const std::string& question_id);

// Throws std::invalid_argument with the question id and the reason: an unknown type, a missing
// criteria block, a schema that does not fit in the budget, or a record that carries media.
[[nodiscard]] Record encode_record(const Request& record, const FragmentEncoder& encoder,
                                   const EncodeRecordOptions& options = {});

} // namespace ninfer::joint_head
