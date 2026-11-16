// The released Clef joint schema head, ported to C++ as an FP32 host implementation.
//
// Ownership note: the head is not part of a `.ninfer` artifact and never enters the backbone's
// operator registry, so it carries its own math instead of borrowing `ops::linear` (whose BF16
// kernels are specialised per exact problem shape). That keeps this file the whole definition of what
// the decision head computes: a direct transcription of `JointSchemaHead.forward` from the release's
// `joint_schema_model.py`, in the same order, with the same reductions.
//
// Two properties of the original are load-bearing and were measured rather than assumed
// (`clef_merge/head_semantics_probe.py`):
//   * both attention stages are NON-causal. `nn.TransformerDecoderLayer` is called with no masks, and
//     adding the causal mask a decoder usually implies moves every multi-question record's answer
//     (max abs difference 1.3 against the module, versus 4.8e-07 for the non-causal reading).
//   * `norm_first` TransformerDecoderLayer applies no output norm, so the final norm is `field_norm`.
//
// The numbers here are an oracle, not a fast path: it is what the device backend has to agree with.

#pragma once

#include "joint_head/record.h"
#include "joint_head/weights.h"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace ninfer::joint_head {

// joint_head_config.json in the release.
struct HeadConfig {
    int hidden_size = 5120;
    int width = 1024;
    int routing_layers = 2;
    int layers = 4;
    int heads = 16;
    int feedforward = 4096;
};

class JointSchemaHead {
public:
    JointSchemaHead(HeadConfig config, const HeadWeights& weights, LexicalTable& lexical);

    // One logit vector per question, in the order the record lists its questions, with one entry per
    // option in the order `QuestionSpec::options` lists them. `hidden` is `tokens * hidden_size`
    // FP32 rows: the post-final-norm backbone state that Engine::export_hidden returns, widened.
    [[nodiscard]] std::vector<std::vector<float>> score(std::span<const float> hidden,
                                                        const Record& record) const;

    [[nodiscard]] const HeadConfig& config() const noexcept { return config_; }
    [[nodiscard]] static constexpr float layer_norm_epsilon() noexcept { return 1e-5F; }

private:
    HeadConfig config_;
    const HeadWeights& weights_;
    LexicalTable& lexical_;
};

} // namespace ninfer::joint_head
