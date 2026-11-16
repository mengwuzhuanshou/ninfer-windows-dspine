// Port of `encode_record` from the release's `joint_schema_model.py`. See encoder.h for the two
// behaviours that are easy to lose: questions iterate in request order, and render() canonicalises
// everything except that outer order.

#include "joint_head/encoder.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::joint_head {
namespace {

constexpr std::string_view system_prompt =
    "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one "
    "of that field's allowed options.";

struct Criterion {
    std::string_view option_id;
    std::string_view description;
};

// The noul defaults: a record may rewrite the wording, never the key set or the order.
constexpr Criterion noul_criteria[] {
    {"true", "The proposition is true or the answer is yes."},
    {"false", "The proposition is false or the answer is no."},
};

struct QuestionTypeEntry {
    std::string_view name;
    QuestionType type;
};

constexpr QuestionTypeEntry question_types[] {
    {"noul", QuestionType::Noul}, {"choice", QuestionType::Choice}, {"score", QuestionType::Score},
};

[[nodiscard]] const Request* member(const Request& object, std::string_view key) {
    // std::string's constructor from string_view is explicit, so the key is named here rather than
    // relying on an implicit conversion at the map's find().
    const auto found = object.find(std::string(key));
    if (found == object.end()) {
        return nullptr;
    }
    return &*found;
}

[[nodiscard]] std::optional<QuestionType> type_of(std::string_view name) {
    for (const auto& entry : question_types) {
        if (entry.name == name) {
            return entry.type;
        }
    }
    return std::nullopt;
}

// `str(question["type"])` in the release. A non-string type would stringify into something that is not
// in QUESTION_TYPES and raise there, so rejecting it here is the same contract with a better message.
[[nodiscard]] std::string type_name(const Request& question, const std::string& question_id) {
    const Request* type = member(question, "type");
    if (type == nullptr) {
        throw std::invalid_argument("question '" + question_id + "' has no type");
    }
    if (!type->is_string()) {
        throw std::invalid_argument("question '" + question_id + "' has a type that is not a string");
    }
    const std::string name = type->get<std::string>();
    if (!type_of(name)) {
        throw std::invalid_argument("question '" + question_id + "' has unknown type '" + name + "'");
    }
    return name;
}

// One fragment appended to the running id list; returns the new length, which is what len() gave the
// release at the same point.
[[nodiscard]] std::size_t push(std::vector<std::int32_t>& into, const FragmentEncoder& encoder,
                               std::string_view fragment) {
    const std::vector<std::int32_t> ids = encoder.encode(fragment);
    into.insert(into.end(), ids.begin(), ids.end());
    return into.size();
}

// The release composes these with f-strings, so every space and newline here is part of the contract.
[[nodiscard]] std::string field_header(std::size_t index, const std::string& question_id,
                                       const std::string& type) {
    std::string text = "\nFIELD ";
    text += std::to_string(index + 1);
    text += "\nID: ";
    text += question_id;
    text += "\nTYPE: ";
    text += type;
    text += "\nINSTRUCTION: ";
    return text;
}

[[nodiscard]] std::string option_header(std::size_t index) {
    return "OPTION " + std::to_string(index + 1) + ": ";
}

// instructions=None or "" falls back to the question id as plain text, which then goes through render()
// as a string and so is emitted unquoted.
[[nodiscard]] std::string instruction_text(const Request& question, const std::string& question_id) {
    const Request* instructions = member(question, "instructions");
    if (instructions == nullptr || instructions->is_null() ||
        (instructions->is_string() && instructions->get<std::string>().empty())) {
        return question_id;
    }
    return render(*instructions);
}

[[nodiscard]] std::string option_semantics_text(const std::string& option_id,
                                                const Request& description) {
    Request semantics = Request::object();
    semantics["option_id"] = option_id;
    if (!description.is_null()) {
        semantics["description"] = description;
    }
    return render(semantics);
}

[[nodiscard]] std::string chat_prefix() {
    std::string text = "<|im_start|>system\n";
    text += std::string(system_prompt);
    text += "<|im_end|>\n<|im_start|>user\nSTATE:\n";
    return text;
}

[[nodiscard]] std::string chat_suffix() {
    std::string text = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    text += "JOINT SCHEMA DECISIONS:";
    return text;
}

// ------------------------------------------------------------------ canonical JSON rendering

void write_escaped(std::string& out, std::string_view text);

void write_number(std::string& out, const Request& value) {
    if (value.is_number_unsigned()) {
        out += std::to_string(value.get<std::uint64_t>());
        return;
    }
    if (value.is_number_integer()) {
        out += std::to_string(value.get<std::int64_t>());
        return;
    }
    // json.dumps and nlohmann both print the shortest round-tripping decimal and both keep a `.0` on an
    // integral float. A non-finite float is where the two genuinely part (Python writes the word NaN), so
    // it is rejected here rather than silently diverging from what the tokenizer would be fed.
    if (!std::isfinite(value.get<double>())) {
        throw std::invalid_argument("render: a state value is not a finite number");
    }
    out += value.dump();
}

void write_value(std::string& out, const Request& value) {
    switch (value.type()) {
    case Request::value_t::null:
        out += "null";
        return;
    case Request::value_t::boolean:
        out += value.get<bool>() ? "true" : "false";
        return;
    case Request::value_t::number_integer:
    case Request::value_t::number_unsigned:
    case Request::value_t::number_float:
        write_number(out, value);
        return;
    case Request::value_t::string:
        out += '"';
        write_escaped(out, value.get_ref<const std::string&>());
        out += '"';
        return;
    case Request::value_t::array: {
        out += '[';
        bool first = true;
        for (const Request& element : value) {
            if (!first) {
                out += ',';
            }
            first = false;
            write_value(out, element);
        }
        out += ']';
        return;
    }
    case Request::value_t::object: {
        // sort_keys=True. This container keeps insertion order, so the keys are collected and sorted by
        // UTF-8 bytes, which is code-point order and therefore what Python's str comparison gives.
        std::vector<std::string> keys;
        keys.reserve(value.size());
        for (auto it = value.begin(); it != value.end(); ++it) {
            keys.push_back(it.key());
        }
        std::sort(keys.begin(), keys.end());
        out += '{';
        for (std::size_t index = 0; index < keys.size(); ++index) {
            if (index != 0) {
                out += ',';
            }
            out += '"';
            write_escaped(out, keys[index]);
            out += "\":";
            write_value(out, value.at(keys[index]));
        }
        out += '}';
        return;
    }
    case Request::value_t::binary:
    case Request::value_t::discarded:
        throw std::invalid_argument("render: a state value is not plain JSON");
    }
    throw std::invalid_argument("render: a state value is not plain JSON");
}

// json.dumps(..., ensure_ascii=False, separators=(",", ":")). These are exactly the escapes Python's
// encoder special-cases; any other control code becomes a \u00XX escape and non-ASCII stays literal.
void write_escaped(std::string& out, std::string_view text) {
    static constexpr char hex[] = "0123456789abcdef";
    for (const char raw : text) {
        const auto code = static_cast<unsigned char>(raw);
        if (code == 0x22) {
            out += "\\\"";
            continue;
        }
        if (code == 0x5C) {
            out += "\\\\";
            continue;
        }
        if (code >= 0x20) {
            out += raw;
            continue;
        }
        switch (code) {
        case 0x08:
            out += "\\b";
            continue;
        case 0x0C:
            out += "\\f";
            continue;
        case 0x0A:
            out += "\\n";
            continue;
        case 0x0D:
            out += "\\r";
            continue;
        case 0x09:
            out += "\\t";
            continue;
        default:
            break;
        }
        out += "\\u00";
        out += hex[(code >> 4) & 0x0F];
        out += hex[code & 0x0F];
    }
}

// ------------------------------------------------------------- option order, per question type

[[nodiscard]] const Request& criteria_of(const Request& question, const std::string& question_id) {
    const Request* criteria = member(question, "criteria");
    if (criteria == nullptr || criteria->is_null()) {
        throw std::invalid_argument("question '" + question_id + "' has no criteria");
    }
    return *criteria;
}

// A noul record may rewrite either wording but never the key set or the order, and the defaults fill in
// whatever it leaves out -- this is `criteria = {...}; criteria.update(question.get("criteria") or {})`.
[[nodiscard]] std::vector<std::pair<std::string, Request>> noul_options(
    const Request& question, const std::string& question_id) {
    Request merged = Request::object();
    const Request* criteria = member(question, "criteria");
    if (criteria != nullptr && !criteria->is_null()) {
        if (!criteria->is_object()) {
            throw std::invalid_argument("a noul question's criteria must be an object (" + question_id +
                                        ")");
        }
        merged = *criteria;
    }
    std::vector<std::pair<std::string, Request>> options;
    options.reserve(2);
    for (const Criterion& criterion : noul_criteria) {
        const Request* replacement = member(merged, criterion.option_id);
        options.emplace_back(std::string(criterion.option_id),
                             replacement == nullptr
                                 ? Request(std::string(criterion.description))
                                 : *replacement);
    }
    return options;
}

} // namespace

std::string render(const Request& value) {
    if (value.is_string()) {
        return value.get<std::string>();
    }
    std::string out;
    write_value(out, value);
    return out;
}

std::vector<std::pair<std::string, Request>> question_options(const Request& question,
                                                              const std::string& question_id) {
    const std::string type = type_name(question, question_id);
    if (type == "noul") {
        return noul_options(question, question_id);
    }
    const Request& criteria = criteria_of(question, question_id);
    std::vector<std::pair<std::string, Request>> options;
    if (type == "choice") {
        if (!criteria.is_object()) {
            throw std::invalid_argument("a choice question's criteria must be an object (" + question_id +
                                        ")");
        }
        for (auto it = criteria.begin(); it != criteria.end(); ++it) {
            options.emplace_back(it.key(), *it);
        }
        // sorted((str(key), value) for key, value in criteria.items()): the key alone decides the order.
        std::stable_sort(options.begin(), options.end(),
                         [](const std::pair<std::string, Request>& left,
                            const std::pair<std::string, Request>& right) {
                             return left.first < right.first;
                         });
        return options;
    }
    if (!criteria.is_array()) {
        throw std::invalid_argument("a score question's criteria must be a list (" + question_id + ")");
    }
    options.reserve(criteria.size());
    for (std::size_t index = 0; index < criteria.size(); ++index) {
        options.emplace_back(std::to_string(index), criteria.at(index));
    }
    return options;
}

Record encode_record(const Request& record, const FragmentEncoder& encoder,
                     const EncodeRecordOptions& options) {
    // Media needs the release processor (patched image/video token runs). This path is text-only, and it
    // rejects the same way run_decision.py's engine branch does.
    if (member(record, "images") != nullptr || member(record, "videos") != nullptr) {
        throw std::invalid_argument(
            "the C++ encoder has no media route yet: this record carries images or videos");
    }
    const Request* questions = member(record, "questions");
    if (questions == nullptr || !questions->is_object() || questions->empty()) {
        throw std::invalid_argument("record carries no questions");
    }

    std::vector<std::int32_t> schema_ids;
    push(schema_ids, encoder, "\n\nSCHEMA FIELDS:\n");
    std::vector<QuestionSpec> specs;
    std::size_t field_index = 0;
    for (auto it = questions->begin(); it != questions->end(); ++it, ++field_index) {
        const std::string question_id = it.key();
        const Request& question = *it;
        const std::string type = type_name(question, question_id);

        push(schema_ids, encoder, field_header(field_index, question_id, type));
        const std::size_t question_start = schema_ids.size();
        push(schema_ids, encoder, instruction_text(question, question_id));
        const std::size_t question_end = schema_ids.size();
        push(schema_ids, encoder, "\nALLOWED OPTIONS:\n");

        QuestionSpec spec;
        spec.question_id = question_id;
        spec.type = *type_of(type);
        spec.question = Span {question_start, question_end};
        std::size_t option_index = 0;
        for (const std::pair<std::string, Request>& option : question_options(question, question_id)) {
            push(schema_ids, encoder, option_header(option_index));
            const std::size_t option_start = schema_ids.size();
            push(schema_ids, encoder, option_semantics_text(option.first, option.second));
            spec.options.push_back(Span {option_start, schema_ids.size()});
            spec.option_ids.push_back(option.first);
            ++option_index;
            push(schema_ids, encoder, "\n");
        }
        if (spec.options.empty()) {
            throw std::invalid_argument("question '" + question_id + "' produced no options");
        }
        push(schema_ids, encoder, "END FIELD\n");
        specs.push_back(std::move(spec));
    }

    std::vector<std::int32_t> prefix_ids;
    push(prefix_ids, encoder, chat_prefix());
    std::vector<std::int32_t> suffix_ids;
    push(suffix_ids, encoder, chat_suffix());

    const Request* state = member(record, "state");
    if (state == nullptr) {
        throw std::invalid_argument("record carries no state");
    }
    std::vector<std::int32_t> state_ids;
    push(state_ids, encoder, render(*state));
    if (options.max_state_tokens) {
        state_ids.resize(std::min(state_ids.size(), *options.max_state_tokens));
    }
    const std::size_t fixed_length = prefix_ids.size() + schema_ids.size() + suffix_ids.size();
    if (fixed_length > options.max_length) {
        throw std::invalid_argument("schema requires " + std::to_string(fixed_length) +
                                    " tokens before state; maximum is " +
                                    std::to_string(options.max_length));
    }
    state_ids.resize(std::min(state_ids.size(), options.max_length - fixed_length));

    // Spans are measured inside the schema block, then shifted by everything that precedes it.
    const std::size_t schema_offset = prefix_ids.size() + state_ids.size();
    for (QuestionSpec& spec : specs) {
        spec.question = Span {spec.question.begin + schema_offset, spec.question.end + schema_offset};
        for (Span& option : spec.options) {
            option = Span {option.begin + schema_offset, option.end + schema_offset};
        }
    }

    Record out;
    out.token_ids = prefix_ids;
    out.token_ids.insert(out.token_ids.end(), state_ids.begin(), state_ids.end());
    out.token_ids.insert(out.token_ids.end(), schema_ids.begin(), schema_ids.end());
    out.token_ids.insert(out.token_ids.end(), suffix_ids.begin(), suffix_ids.end());
    out.questions = std::move(specs);
    if (out.token_ids.empty() || out.questions.empty()) {
        throw std::invalid_argument("record produced no model input or questions");
    }
    return out;
}

} // namespace ninfer::joint_head
