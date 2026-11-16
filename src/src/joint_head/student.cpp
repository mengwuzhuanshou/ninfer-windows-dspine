#include "joint_head/student.h"

#include "joint_head/linalg.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::joint_head {
namespace {

using namespace detail;

// exp() of the stored parameter, capped the way the trainer caps it (`log_logit_scale.clamp(max =
// log(100))`). The clamp is load-bearing: an uncapped temperature turns a near-tie into a one-hot.
[[nodiscard]] float temperature(float log_scale) noexcept {
    return std::exp(std::min(log_scale, std::log(100.0F)));
}

[[nodiscard]] Mat normalize_rows(const Mat& source) {
    Mat out = matrix(source.rows, source.cols);
    for (std::size_t row = 0; row < source.rows; ++row) {
        const std::vector<float> unit =
            l2_normalize(std::span<const float>(source.at(row), source.cols));
        std::memcpy(out.at(row), unit.data(), source.cols * sizeof(float));
    }
    return out;
}

[[nodiscard]] BiasView bias_of(const Tensor& tensor) {
    return BiasView(tensor.values.data(), tensor.values.size());
}

class Forward {
public:
    Forward(const StudentConfig& config, const HeadWeights& weights, LexicalTable& lexical)
        : config_(config), weights_(weights), lexical_(lexical) {}

    std::vector<std::vector<float>> run(std::span<const float> hidden, const Record& record) const;

private:
    [[nodiscard]] const Tensor& tensor(std::string_view name) const { return weights_.tensor(name); }

    StudentConfig config_;
    const HeadWeights& weights_;
    LexicalTable& lexical_;
};

std::vector<std::vector<float>> Forward::run(std::span<const float> hidden,
                                             const Record& record) const {
    const std::size_t hidden_size = config_.hidden_size;
    const std::size_t width = config_.width;
    if (record.questions.empty()) {
        throw std::invalid_argument("student head: the record carries no questions");
    }
    if (hidden.size() != record.tokens() * hidden_size) {
        throw std::invalid_argument("student head: got " +
                                    std::to_string(hidden.size() / hidden_size) + " hidden rows for a "
                                    "record of " + std::to_string(record.tokens()) + " tokens");
    }

    const Tensor& type_embedding = tensor("type_embedding.weight");
    if (static_cast<std::size_t>(type_embedding.dims[0]) < config_.types) {
        throw std::invalid_argument("student head: type embedding has " +
                                    std::to_string(type_embedding.dims[0]) + " rows, the config needs " +
                                    std::to_string(config_.types));
    }

    // Raw pooling: no hidden_norm anywhere in this path, see the header.
    const Mat sequence = rows_of(hidden, hidden_size);
    std::vector<Span> question_spans;
    question_spans.reserve(record.questions.size());
    for (const QuestionSpec& question : record.questions) {
        question_spans.push_back(question.question);
    }
    const Mat question_vectors = mean_rows(sequence, question_spans, "question");
    const float* global_row = sequence.at(record.tokens() - 1);

    // state_in = [question_pooled ; last hidden token of the record], one row per question.
    Mat state_in = matrix(record.questions.size(), 2 * hidden_size);
    for (std::size_t index = 0; index < record.questions.size(); ++index) {
        float* target = state_in.at(index);
        std::memcpy(target, question_vectors.at(index), hidden_size * sizeof(float));
        std::memcpy(target + hidden_size, global_row, hidden_size * sizeof(float));
    }

    Mat queries = linear(state_in, weight_view(tensor("state.0.weight"), "state.0"),
                         bias_of(tensor("state.0.bias")));
    for (float& value : queries.data) {
        value = gelu(value);
    }
    queries = linear(queries, weight_view(tensor("state.2.weight"), "state.2"),
                     bias_of(tensor("state.2.bias")));
    const float* types = type_embedding.values.data();
    for (std::size_t index = 0; index < queries.rows; ++index) {
        const std::int32_t type = static_cast<std::int32_t>(record.questions[index].type);
        if (type < 0 || static_cast<std::size_t>(type) >= static_cast<std::size_t>(type_embedding.dims[0])) {
            throw std::invalid_argument("student head: question '" +
                                        record.questions[index].question_id + "' has type " +
                                        std::to_string(type));
        }
        const float* embed = types + static_cast<std::size_t>(type) * width;
        float* target = queries.at(index);
        for (std::size_t column = 0; column < width; ++column) {
            target[column] += embed[column];
        }
    }
    const Mat states = normalize_rows(queries);

    const WeightView context_projection =
        weight_view(tensor("option_context.weight"), "option_context");
    const WeightView lexical_projection =
        weight_view(tensor("option_lexical.weight"), "option_lexical");
    const WeightView option_projection = weight_view(tensor("option.1.weight"), "option.1");
    const BiasView context_bias = bias_of(tensor("option_context.bias"));
    const BiasView lexical_bias = bias_of(tensor("option_lexical.bias"));
    const BiasView option_bias = bias_of(tensor("option.1.bias"));

    const float scale = temperature(weights_.scalar("log_logit_scale"));
    std::vector<std::vector<float>> logits;
    logits.reserve(record.questions.size());
    for (std::size_t index = 0; index < record.questions.size(); ++index) {
        const QuestionSpec& question = record.questions[index];
        if (question.option_ids.size() != question.options.size()) {
            throw std::invalid_argument("student head: question '" + question.question_id + "' lists " +
                                        std::to_string(question.options.size()) + " spans for " +
                                        std::to_string(question.option_ids.size()) + " option ids");
        }
        Mat options = mean_rows(sequence, question.options, "option");

        Mat lexicon = matrix(question.options.size(), hidden_size);
        for (std::size_t row = 0; row < lexicon.rows; ++row) {
            const std::vector<float> lexical_row =
                mean_lexical(lexical_, record.token_ids, question.options[row], hidden_size,
                             "option lexical");
            std::memcpy(lexicon.at(row), lexical_row.data(), hidden_size * sizeof(float));
        }

        Mat scored = linear(options, context_projection, context_bias);
        add_rows(linear(lexicon, lexical_projection, lexical_bias), scored);
        for (float& value : scored.data) {
            value = gelu(value);
        }
        const Mat keys = normalize_rows(linear(scored, option_projection, option_bias));

        const float* state = states.at(index);
        std::vector<float> answer(keys.rows);
        for (std::size_t row = 0; row < keys.rows; ++row) {
            answer[row] = scale * dot(state, keys.at(row), width);
        }
        logits.push_back(std::move(answer));
    }
    return logits;
}

} // namespace

StudentHead::StudentHead(StudentConfig config, const HeadWeights& weights, LexicalTable& lexical)
    : config_(config), weights_(weights), lexical_(lexical) {
    if (config_.hidden_size == 0 || config_.width == 0 || config_.types == 0) {
        throw std::invalid_argument("student head: the head config has a zero stage size");
    }
    validate_student_shapes(weights, config);
}

std::vector<std::vector<float>> StudentHead::score(std::span<const float> hidden,
                                                   const Record& record) const {
    return Forward(config_, weights_, lexical_).run(hidden, record);
}

float StudentHead::logit_scale() const {
    // Not noexcept: scalar() can throw for a malformed sidecar, and the cap is part of the number.
    return std::exp(std::min(weights_.scalar("log_logit_scale"), std::log(100.0F)));
}

} // namespace ninfer::joint_head
