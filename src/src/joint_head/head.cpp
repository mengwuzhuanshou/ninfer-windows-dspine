#include "joint_head/head.h"
#include "joint_head/linalg.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::joint_head {
namespace {

// The FP32 building blocks this file computes with are shared with the student head and live
// in joint_head/linalg.h. They are imported unqualified so the transcription below stays a
// line-by-line read of JointSchemaHead.forward in the release module.
using namespace detail;

// --------------------------------------------------------------------------- attention

// nn.MultiheadAttention(width, heads, batch_first=True) with separate query and key/value sources and
// no mask: NON-causal in both directions, which is what the release module actually computes here.
[[nodiscard]] Mat multi_head_attention(const Mat& query, const Mat& key, const Mat& value,
                                       const Tensor& in_proj_weight, const Tensor& in_proj_bias,
                                       const Tensor& out_proj_weight, const Tensor& out_proj_bias,
                                       std::size_t heads) {
    const std::size_t width = query.cols;
    if (key.cols != width || value.cols != width) {
        throw std::invalid_argument("joint head: attention key/value width differs from query width");
    }
    if (heads == 0 || width % heads != 0) {
        throw std::invalid_argument("joint head: width " + std::to_string(width) +
                                    " does not split into " + std::to_string(heads) + " heads");
    }
    const WeightView projections = weight_view(in_proj_weight, "attention.in_proj_weight");
    if (projections.rows != 3 * width || projections.cols != width) {
        throw std::invalid_argument("joint head: attention in_proj is " +
                                    std::to_string(projections.rows) + "x" +
                                    std::to_string(projections.cols) + ", expected 3x" +
                                    std::to_string(width) + "x" + std::to_string(width));
    }
    const BiasView bias(in_proj_bias.values.data(), in_proj_bias.values.size());
    const Mat queries = linear(query, projections.block(0, width), bias.subspan(0, width));
    const Mat keys = linear(key, projections.block(width, width), bias.subspan(width, width));
    const Mat values = linear(value, projections.block(2 * width, width),
                              bias.subspan(2 * width, width));

    const std::size_t head_dim = width / heads;
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    Mat context = matrix(query.rows, width);
    std::vector<float> scores(key.rows);
    for (std::size_t head = 0; head < heads; ++head) {
        const std::size_t offset = head * head_dim;
        for (std::size_t row = 0; row < query.rows; ++row) {
            const float* query_row = queries.at(row) + offset;
            for (std::size_t column = 0; column < key.rows; ++column) {
                scores[column] = dot(query_row, keys.at(column) + offset, head_dim) * scale;
            }
            softmax_inplace(scores.data(), scores.size());
            float* target = context.at(row) + offset;
            for (std::size_t column = 0; column < key.rows; ++column) {
                const float weight = scores[column];
                const float* value_row = values.at(column) + offset;
                for (std::size_t axis = 0; axis < head_dim; ++axis) {
                    target[axis] += weight * value_row[axis];
                }
            }
        }
    }
    return linear(context, weight_view(out_proj_weight, "attention.out_proj.weight"),
                  BiasView(out_proj_bias.values.data(), out_proj_bias.values.size()));
}

// --------------------------------------------------------------------------- the forward

class Forward {
public:
    Forward(const HeadConfig& config, const HeadWeights& weights, LexicalTable& lexical)
        : config_(config),
          weights_(weights),
          lexical_(lexical),
          width_(static_cast<std::size_t>(config.width)),
          hidden_size_(static_cast<std::size_t>(config.hidden_size)),
          heads_(static_cast<std::size_t>(config.heads)) {}

    std::vector<std::vector<float>> run(std::span<const float> hidden, const Record& record) const;

private:
    [[nodiscard]] const Tensor& tensor(std::string_view name) const {
        return weights_.tensor(name);
    }
    [[nodiscard]] std::string field(std::string_view prefix, std::string_view suffix) const {
        return std::string(prefix) + std::string(suffix);
    }

    [[nodiscard]] Mat evidence_layer(const Mat& queries, const Mat& memory, int index) const;
    [[nodiscard]] Mat decoder_layer(const Mat& fields, const Mat& memory, int index) const;

    HeadConfig config_;
    const HeadWeights& weights_;
    LexicalTable& lexical_;
    std::size_t width_ = 0;
    std::size_t hidden_size_ = 0;
    std::size_t heads_ = 0;
};

Mat Forward::evidence_layer(const Mat& queries, const Mat& memory, int index) const {
    const std::string prefix = "evidence_layers." + std::to_string(index) + ".";
    Mat normalised_queries = layer_norm(queries, tensor(field(prefix, "query_norm.weight")),
                                       tensor(field(prefix, "query_norm.bias")),
                                       JointSchemaHead::layer_norm_epsilon());
    Mat normalised_memory =
        layer_norm(memory, tensor(field(prefix, "memory_norm.weight")),
                   tensor(field(prefix, "memory_norm.bias")),
                   JointSchemaHead::layer_norm_epsilon());
    Mat routed = multi_head_attention(
        normalised_queries, normalised_memory, normalised_memory,
        tensor(field(prefix, "attention.in_proj_weight")),
        tensor(field(prefix, "attention.in_proj_bias")),
        tensor(field(prefix, "attention.out_proj.weight")),
        tensor(field(prefix, "attention.out_proj.bias")), heads_);

    Mat accumulated = queries;
    add_rows(routed, accumulated);
    Mat normalised = layer_norm(accumulated, tensor(field(prefix, "feedforward_norm.weight")),
                               tensor(field(prefix, "feedforward_norm.bias")),
                               JointSchemaHead::layer_norm_epsilon());
    const Tensor& up_weight = tensor(field(prefix, "feedforward.0.weight"));
    const Tensor& up_bias = tensor(field(prefix, "feedforward.0.bias"));
    const Tensor& down_weight = tensor(field(prefix, "feedforward.3.weight"));
    const Tensor& down_bias = tensor(field(prefix, "feedforward.3.bias"));
    Mat hidden = linear(normalised, weight_view(up_weight, "feedforward.0"),
                        BiasView(up_bias.values.data(), up_bias.values.size()));
    for (float& value : hidden.data) {
        value = gelu(value);
    }
    Mat driven = linear(hidden, weight_view(down_weight, "feedforward.3"),
                        BiasView(down_bias.values.data(), down_bias.values.size()));
    add_rows(driven, accumulated);
    return accumulated;
}

Mat Forward::decoder_layer(const Mat& fields, const Mat& memory, int index) const {
    const std::string prefix = "layers." + std::to_string(index) + ".";
    const float epsilon = JointSchemaHead::layer_norm_epsilon();

    Mat output = fields;
    Mat self_input = layer_norm(output, tensor(field(prefix, "norm1.weight")),
                                tensor(field(prefix, "norm1.bias")), epsilon);
    Mat self_attended = multi_head_attention(
        self_input, self_input, self_input, tensor(field(prefix, "self_attn.in_proj_weight")),
        tensor(field(prefix, "self_attn.in_proj_bias")),
        tensor(field(prefix, "self_attn.out_proj.weight")),
        tensor(field(prefix, "self_attn.out_proj.bias")), heads_);
    add_rows(self_attended, output);

    Mat cross_input = layer_norm(output, tensor(field(prefix, "norm2.weight")),
                                 tensor(field(prefix, "norm2.bias")), epsilon);
    // Memory reaches the decoder's cross attention raw: TransformerDecoderLayer owns no memory norm,
    // which is what the hand-built layer in head_semantics_probe.py confirmed against the module.
    Mat cross_attended = multi_head_attention(
        cross_input, memory, memory, tensor(field(prefix, "multihead_attn.in_proj_weight")),
        tensor(field(prefix, "multihead_attn.in_proj_bias")),
        tensor(field(prefix, "multihead_attn.out_proj.weight")),
        tensor(field(prefix, "multihead_attn.out_proj.bias")), heads_);
    add_rows(cross_attended, output);

    Mat gate = layer_norm(output, tensor(field(prefix, "norm3.weight")),
                          tensor(field(prefix, "norm3.bias")), epsilon);
    const Tensor& up_weight = tensor(field(prefix, "linear1.weight"));
    const Tensor& up_bias = tensor(field(prefix, "linear1.bias"));
    const Tensor& down_weight = tensor(field(prefix, "linear2.weight"));
    const Tensor& down_bias = tensor(field(prefix, "linear2.bias"));
    Mat raised = linear(gate, weight_view(up_weight, "linear1"),
                        BiasView(up_bias.values.data(), up_bias.values.size()));
    for (float& value : raised.data) {
        value = gelu(value);
    }
    Mat lowered = linear(raised, weight_view(down_weight, "linear2"),
                         BiasView(down_bias.values.data(), down_bias.values.size()));
    add_rows(lowered, output);
    return output;
}

std::vector<std::vector<float>> Forward::run(std::span<const float> hidden,
                                             const Record& record) const {
    const float epsilon = JointSchemaHead::layer_norm_epsilon();
    if (record.questions.empty()) {
        throw std::invalid_argument("joint head: the record carries no questions");
    }
    if (hidden.size() != record.tokens() * hidden_size_) {
        throw std::invalid_argument("joint head: got " + std::to_string(hidden.size() / hidden_size_) +
                                    " hidden rows for a record of " +
                                    std::to_string(record.tokens()) + " tokens");
    }

    // normalized_hidden / memory / global_vector
    Mat sequence = rows_of(hidden, hidden_size_);
    Mat normalised = layer_norm(sequence, tensor("hidden_norm.weight"), tensor("hidden_norm.bias"),
                                epsilon);
    Mat memory = linear(normalised, weight_view(tensor("memory_projection.weight"),
                                                "memory_projection"),
                        {});
    std::vector<float> global_vector(normalised.at(record.tokens() - 1),
                                     normalised.at(record.tokens() - 1) + hidden_size_);

    std::vector<Span> question_spans;
    question_spans.reserve(record.questions.size());
    for (const QuestionSpec& question : record.questions) {
        question_spans.push_back(question.question);
    }
    Mat question_vectors = mean_rows(normalised, question_spans, "question");

    // Per-question option context (mean hidden over the option span) and option lexicon (mean lm_head
    // row over the same tokens). Both stay in hidden_size space until their own projections.
    std::vector<Mat> option_contexts;
    std::vector<std::vector<std::vector<float>>> option_lexicons;
    option_contexts.reserve(record.questions.size());
    option_lexicons.reserve(record.questions.size());
    for (const QuestionSpec& question : record.questions) {
        option_contexts.push_back(mean_rows(normalised, question.options, "option"));
        std::vector<std::vector<float>> lexicons;
        lexicons.reserve(question.options.size());
        for (const Span option : question.options) {
            lexicons.push_back(mean_lexical(lexical_, record.token_ids, option, hidden_size_,
                                            "option lexical"));
        }
        option_lexicons.push_back(std::move(lexicons));
    }

    // option_queries, concatenated in question order, routed against memory, split again.
    Mat queries = matrix(0, width_);
    std::vector<std::size_t> option_counts;
    option_counts.reserve(record.questions.size());
    const WeightView context_projection =
        weight_view(tensor("option_context_projection.weight"), "option_context_projection");
    const WeightView lexical_projection =
        weight_view(tensor("option_lexical_projection.weight"), "option_lexical_projection");
    const WeightView option_question_projection =
        weight_view(tensor("option_question_projection.weight"), "option_question_projection");
    for (std::size_t index = 0; index < record.questions.size(); ++index) {
        const Mat& context = option_contexts[index];
        const Mat lexicon = [&] {
            Mat built = matrix(option_lexicons[index].size(), hidden_size_);
            for (std::size_t row = 0; row < built.rows; ++row) {
                std::memcpy(built.at(row), option_lexicons[index][row].data(),
                            hidden_size_ * sizeof(float));
            }
            return built;
        }();
        Mat routed_block = linear(context, context_projection, {});
        add_rows(linear(lexicon, lexical_projection, {}), routed_block);
        std::vector<float> question_row =
            linear_one(question_vectors.at(index), option_question_projection, {});
        for (std::size_t row = 0; row < routed_block.rows; ++row) {
            for (std::size_t column = 0; column < width_; ++column) {
                routed_block.at(row)[column] += question_row[column];
            }
        }
        queries.rows += routed_block.rows;
        queries.data.insert(queries.data.end(), routed_block.data.begin(), routed_block.data.end());
        option_counts.push_back(routed_block.rows);
    }

    Mat routed_options = queries;
    for (int layer = 0; layer < config_.routing_layers; ++layer) {
        routed_options = evidence_layer(routed_options, memory, layer);
    }
    std::vector<Mat> split_options;
    split_options.reserve(option_counts.size());
    {
        std::size_t cursor = 0;
        for (const std::size_t count : option_counts) {
            Mat part = matrix(count, width_);
            for (std::size_t row = 0; row < count; ++row) {
                std::memcpy(part.at(row), routed_options.at(cursor + row), width_ * sizeof(float));
            }
            cursor += count;
            split_options.push_back(std::move(part));
        }
        if (cursor != routed_options.rows) {
            throw std::logic_error("joint head: option split lost rows");
        }
    }

    // Field queries: the projected question vector, plus a norm-bounded summary of the options that
    // attended to it, plus the record-level state and the question type.
    Mat base_fields = linear(question_vectors,
                             weight_view(tensor("question_projection.weight"), "question_projection"),
                             {});
    const Tensor& summary_weight = tensor("option_summary_norm.weight");
    const Tensor& summary_bias = tensor("option_summary_norm.bias");
    Mat summaries = matrix(base_fields.rows, width_);
    const float routing_scale = 1.0F / std::sqrt(static_cast<float>(width_));
    for (std::size_t index = 0; index < split_options.size(); ++index) {
        const Mat& options = split_options[index];
        const float* field = base_fields.at(index);
        std::vector<float> attention(options.rows);
        for (std::size_t row = 0; row < options.rows; ++row) {
            attention[row] = dot(options.at(row), field, width_) * routing_scale;
        }
        softmax_inplace(attention.data(), attention.size());
        float* target = summaries.at(index);
        for (std::size_t column = 0; column < width_; ++column) {
            target[column] = 0.0F;
        }
        for (std::size_t row = 0; row < options.rows; ++row) {
            for (std::size_t column = 0; column < width_; ++column) {
                target[column] += attention[row] * options.at(row)[column];
            }
        }
    }
    Mat normalised_summaries = layer_norm(summaries, summary_weight, summary_bias, epsilon);

    const WeightView global_projection =
        weight_view(tensor("global_projection.weight"), "global_projection");
    const std::vector<float> global_state = linear_one(global_vector.data(), global_projection, {});
    const Tensor& type_embedding = tensor("type_embedding.weight");
    const std::size_t type_width = static_cast<std::size_t>(type_embedding.dims[1]);
    if (type_width != width_) {
        throw std::invalid_argument("joint head: type embedding is " + std::to_string(type_width) +
                                    " wide, the head runs at " + std::to_string(width_));
    }
    Mat fields = base_fields;
    for (std::size_t row = 0; row < fields.rows; ++row) {
        const std::int32_t type = static_cast<std::int32_t>(record.questions[row].type);
        if (type < 0 || static_cast<std::size_t>(type) >= static_cast<std::size_t>(type_embedding.dims[0])) {
            throw std::invalid_argument("joint head: question '" +
                                        record.questions[row].question_id + "' has type " +
                                        std::to_string(type));
        }
        const float* embed = type_embedding.values.data() +
                             static_cast<std::size_t>(type) * type_width;
        for (std::size_t column = 0; column < width_; ++column) {
            fields.at(row)[column] += normalised_summaries.at(row)[column] + global_state[column] +
                                      embed[column];
        }
    }
    for (int layer = 0; layer < config_.layers; ++layer) {
        fields = decoder_layer(fields, memory, layer);
    }
    fields = layer_norm(fields, tensor("field_norm.weight"), tensor("field_norm.bias"), epsilon);

    // Scoring: a lexical prior over the option wording plus the joint term, gated by residual_gate.
    const float prior_scale =
        std::exp(std::min(weights_.scalar("prior_logit_scale"), std::log(100.0F)));
    const float joint_scale =
        std::exp(std::min(weights_.scalar("joint_logit_scale"), std::log(100.0F)));
    const float gate = 1.0F / (1.0F + std::exp(-weights_.scalar("residual_gate")));
    const Tensor& option_norm_weight = tensor("option_norm.weight");
    const Tensor& option_norm_bias = tensor("option_norm.bias");
    const WeightView scorer_up = weight_view(tensor("residual_scorer.0.weight"), "residual_scorer.0");
    const BiasView scorer_up_bias(tensor("residual_scorer.0.bias").values.data(), width_);
    const WeightView scorer_down = weight_view(tensor("residual_scorer.3.weight"),
                                               "residual_scorer.3");
    const BiasView scorer_down_bias(tensor("residual_scorer.3.bias").values.data(), 1);

    std::vector<std::vector<float>> logits;
    logits.reserve(record.questions.size());
    for (std::size_t index = 0; index < record.questions.size(); ++index) {
        const QuestionSpec& question = record.questions[index];
        const Mat& options_routed = split_options[index];
        if (options_routed.rows != question.options.size() ||
            question.option_ids.size() != question.options.size()) {
            throw std::invalid_argument("joint head: question '" + question.question_id +
                                        "' lists " + std::to_string(question.options.size()) +
                                        " spans for " + std::to_string(question.option_ids.size()) +
                                        " option ids");
        }

        std::vector<float> anchor_input(hidden_size_);
        for (std::size_t column = 0; column < hidden_size_; ++column) {
            anchor_input[column] = question_vectors.at(index)[column] + global_vector[column];
        }
        const std::vector<float> anchor = l2_normalize(anchor_input);

        std::vector<float> prior(options_routed.rows);
        Mat lexicon_normalised = matrix(options_routed.rows, hidden_size_);
        for (std::size_t row = 0; row < lexicon_normalised.rows; ++row) {
            const std::vector<float> unit = l2_normalize(option_lexicons[index][row]);
            std::memcpy(lexicon_normalised.at(row), unit.data(), hidden_size_ * sizeof(float));
            prior[row] = prior_scale * dot(unit.data(), anchor.data(), hidden_size_);
        }

        Mat options = layer_norm(options_routed, option_norm_weight, option_norm_bias, epsilon);
        const float* field = fields.at(index);
        Mat features = matrix(options.rows, 4 * width_);
        std::vector<float> cosine(options.rows);
        for (std::size_t row = 0; row < options.rows; ++row) {
            double inner = 0.0;
            double field_square = 0.0;
            double option_square = 0.0;
            float* feature = features.at(row);
            for (std::size_t column = 0; column < width_; ++column) {
                const float left = field[column];
                const float right = options.at(row)[column];
                feature[column] = left;
                feature[width_ + column] = right;
                feature[2 * width_ + column] = left * right;
                feature[3 * width_ + column] = std::fabs(left - right);
                inner += static_cast<double>(left) * static_cast<double>(right);
                field_square += static_cast<double>(left) * static_cast<double>(left);
                option_square += static_cast<double>(right) * static_cast<double>(right);
            }
            const double denominator =
                std::max(std::sqrt(field_square), 1e-12) * std::max(std::sqrt(option_square), 1e-12);
            cosine[row] = static_cast<float>(inner / denominator);
        }
        Mat raised = linear(features, scorer_up, scorer_up_bias);
        for (float& value : raised.data) {
            value = gelu(value);
        }
        const Mat driven = linear(raised, scorer_down, scorer_down_bias);
        std::vector<float> answer(options.rows);
        for (std::size_t row = 0; row < options.rows; ++row) {
            const float residual = driven.at(row)[0];
            const float joint = joint_scale * cosine[row] + residual;
            answer[row] = prior[row] + gate * joint;
        }
        logits.push_back(std::move(answer));
    }
    return logits;
}

} // namespace

JointSchemaHead::JointSchemaHead(HeadConfig config, const HeadWeights& weights,
                                 LexicalTable& lexical)
    : config_(config), weights_(weights), lexical_(lexical) {
    if (config.hidden_size <= 0 || config.width <= 0 || config.heads <= 0 ||
        config.feedforward <= 0 || config.routing_layers < 0 || config.layers < 0) {
        throw std::invalid_argument("joint head: the head config has a non-positive stage size");
    }
    validate_head_shapes(weights, config.hidden_size, config.width, config.heads,
                         config.feedforward, config.routing_layers, config.layers);
}

std::vector<std::vector<float>> JointSchemaHead::score(std::span<const float> hidden,
                                                       const Record& record) const {
    return Forward(config_, weights_, lexical_).run(hidden, record);
}

} // namespace ninfer::joint_head
