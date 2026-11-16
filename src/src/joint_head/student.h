// The CLM-style student head: two projection MLPs over the *frozen* tower, replacing the 128M
// JointSchemaHead for the decision path.
//
// Trained in `clef_merge/head_train/clm_head.py` + `train_clm_head.py` by distilling the released
// head's own per-question probabilities and argmax (KL + hard CE + bidirectional InfoNCE) on stored
// hidden states; the tower never moves and the head is read from a NINHHEAD sidecar exactly like
// `JointSchemaHead` is, so this file is the whole definition of what the student computes.
//
// One difference from the released head is load-bearing and easy to lose: the student pools **raw**
// hidden rows. `JointSchemaHead::score` starts with `hidden_norm` and every downstream vector is in
// normalised space; the student's `extract_features` takes `_mean_span(hidden, ...)` and `hidden[-1]`
// of the un-normalised export directly. A port that "tidied" that norm in here would score fine on
// synthetic inputs and disagree with every real record.
//
// Math, in the order the Python runs it:
//   state_in = [mean(question_span) ; hidden[-1]]                     (2 * hidden_size)
//   q        = normalize(state.2(GELU(state.0(state_in))) + type_embedding[type])
//   o_in     = option_context(mean(option_span)) + option_lexical(mean lm_head row of the option)
//   k        = normalize(option.1(GELU(o_in)))
//   logit    = exp(min(log_logit_scale, log 100)) * (q . k)
//
// Like the released head this is an oracle, not a fast path: it is FP32 host arithmetic written to be
// compared against the module, and the numbers the device backend has to reproduce.

#pragma once

#include "joint_head/record.h"
#include "joint_head/weights.h"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace ninfer::joint_head {

// The trainer's defaults, recorded in `student.<source>.<arm>.pt` next to the state dict.
struct StudentConfig {
    std::size_t hidden_size = 5120;
    std::size_t width = 768;
    std::size_t types = 3;
};

class StudentHead {
public:
    StudentHead(StudentConfig config, const HeadWeights& weights, LexicalTable& lexical);

    // One logit vector per question, in record order, with one entry per option in the order
    // `QuestionSpec::option_ids` lists them -- the same contract as `JointSchemaHead::score`, so a
    // caller can swap one for the other.
    [[nodiscard]] std::vector<std::vector<float>> score(std::span<const float> hidden,
                                                        const Record& record) const;

    [[nodiscard]] const StudentConfig& config() const noexcept { return config_; }
    // The temperature actually used: exp of the stored parameter, clamped the way CLIP clamps it. Can
    // throw -- a sidecar without the parameter is a wrong file, and the clamp is part of the number.
    [[nodiscard]] float logit_scale() const;

private:
    StudentConfig config_;
    const HeadWeights& weights_;
    LexicalTable& lexical_;
};

// Every tensor the forward multiplies against, checked before any of them run.
void validate_student_shapes(const HeadWeights& weights, const StudentConfig& config);

} // namespace ninfer::joint_head
