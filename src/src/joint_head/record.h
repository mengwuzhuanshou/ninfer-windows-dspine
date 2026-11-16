// The record the joint schema head consumes: one token sequence and the question/option spans inside
// it. `encode_record` in the release's `joint_schema_model.py` produces this from a Jev/SystemOne
// request; `joint_head/encoder.cpp` is the port of that function, and the selftest fixtures carry
// spans produced by the Python original so the head's arithmetic can be checked without trusting the
// C++ encoder.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::joint_head {

// Half-open token range [begin, end) inside Record::token_ids.
struct Span {
    std::size_t begin = 0;
    std::size_t end = 0;

    [[nodiscard]] std::size_t length() const noexcept { return end - begin; }
};

// QUESTION_TYPES in the release module: the head's type embedding indexes these rows.
enum class QuestionType : std::int32_t { Noul = 0, Choice = 1, Score = 2 };

struct QuestionSpec {
    std::string question_id;
    QuestionType type = QuestionType::Choice;
    Span question {};
    std::vector<Span> options {};
    std::vector<std::string> option_ids {};
};

struct Record {
    std::vector<std::int32_t> token_ids;
    std::vector<QuestionSpec> questions;

    [[nodiscard]] std::size_t tokens() const noexcept { return token_ids.size(); }
};

} // namespace ninfer::joint_head
