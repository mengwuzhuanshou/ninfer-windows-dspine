// ninfer-decide-serve: the decision head inside the engine process.
//
// This file is the offline half. `--selftest` replays stored hidden-state dumps through the C++ head
// and compares against the Python gold, and `--student-selftest` does the same for the CLM-style
// student head while timing the two against each other. Neither needs a GPU, an artifact load or a
// server: the parity checks are not confounded by the CUDA path.
// `--serve` is handled by `serve.cpp`, which puts the same two heads behind an HTTP route on top of a
// real engine, so the latency the decision path costs is measured in the process that serves it.

#include "joint_head/bf16.h"
#include "joint_head/encoder.h"
#include "joint_head/head.h"
#include "joint_head/record.h"
#include "joint_head/student.h"
#include "joint_head/weights.h"
#include "models/qwen3_5/frontend/tokenizer.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::decide {
// serve.cpp owns the --serve route and takes the whole argv, so its flags stay its own and this file
// stays the offline half.
int serve_main(int argc, char** argv);
} // namespace ninfer::decide

namespace {

using Json = nlohmann::json;
using namespace ninfer::joint_head;

// The engine's BPE frontend, handed to the head module through the one method it needs.
class EngineFragmentEncoder final : public FragmentEncoder {
public:
    explicit EngineFragmentEncoder(ninfer::models::qwen3_5::frontend::Tokenizer& tokenizer)
        : tokenizer_(tokenizer) {}

    [[nodiscard]] std::vector<std::int32_t> encode(std::string_view fragment) const override {
        const std::vector<int> ids = tokenizer_.encode(fragment);
        return std::vector<std::int32_t>(ids.begin(), ids.end());
    }

private:
    ninfer::models::qwen3_5::frontend::Tokenizer& tokenizer_;
};

struct Options {
    std::string mode;
    std::string fixtures;
    std::string prompts;
    std::string release;
    float max_abs_diff = 0.0F;  // 0 means "report, do not gate"
};

[[nodiscard]] std::string read_text(const std::filesystem::path& path) {
    const std::vector<std::uint8_t> image = [&] {
        std::FILE* handle = std::fopen(path.string().c_str(), "rb");
        if (handle == nullptr) {
            throw std::invalid_argument("cannot open " + path.string());
        }
        struct Close {
            std::FILE* handle;
            ~Close() { std::fclose(handle); }
        } closer {handle};
        std::vector<std::uint8_t> bytes;
        bytes.resize(static_cast<std::size_t>(1) << 16);
        std::size_t total = 0;
        for (;;) {
            if (total == bytes.size()) {
                bytes.resize(total * 2);
            }
            const std::size_t read = std::fread(bytes.data() + total, 1, bytes.size() - total, handle);
            if (read == 0) {
                break;
            }
            total += read;
        }
        bytes.resize(total);
        return bytes;
    }();
    return std::string(reinterpret_cast<const char*>(image.data()), image.size());
}

[[nodiscard]] std::vector<std::uint8_t> read_file(const std::filesystem::path& path,
                                                  std::size_t expected) {
    std::FILE* handle = std::fopen(path.string().c_str(), "rb");
    if (handle == nullptr) {
        throw std::invalid_argument("cannot open " + path.string());
    }
    std::vector<std::uint8_t> image(expected);
    const std::size_t read = std::fread(image.data(), 1, image.size(), handle);
    std::fclose(handle);
    if (read != image.size()) {
        throw std::invalid_argument("short read on " + path.string() + " (" + std::to_string(read) +
                                    " of " + std::to_string(image.size()) + " bytes)");
    }
    return image;
}

[[nodiscard]] std::vector<std::int32_t> read_ids(const std::filesystem::path& path,
                                                 std::size_t tokens) {
    const std::vector<std::uint8_t> image = read_file(path, tokens * sizeof(std::int32_t));
    std::vector<std::int32_t> ids(tokens);
    std::memcpy(ids.data(), image.data(), image.size());
    return ids;
}

[[nodiscard]] std::vector<float> read_hidden(const std::filesystem::path& path, std::size_t rows,
                                             std::size_t width) {
    const std::vector<std::uint8_t> image = read_file(path, rows * width * sizeof(std::uint16_t));
    std::vector<std::uint16_t> bits(rows * width);
    std::memcpy(bits.data(), image.data(), image.size());
    return widen(std::span<const std::uint16_t>(bits));
}

[[nodiscard]] Span span_of(const Json& value, const std::string& case_name,
                           const std::string& what) {
    if (!value.is_array() || value.size() != 2) {
        throw std::invalid_argument("case '" + case_name + "': " + what + " is not a [begin, end) pair");
    }
    return Span {value[0].get<std::size_t>(), value[1].get<std::size_t>()};
}

[[nodiscard]] Record record_from_manifest(const Json& entry, const std::filesystem::path& directory,
                                          std::size_t tokens, std::size_t hidden_size) {
    const std::string name = entry.at("name").get<std::string>();
    if (entry.at("tokens").get<std::size_t>() != tokens) {
        throw std::invalid_argument("case '" + name + "': manifest declares " +
                                    std::to_string(entry.at("tokens").get<std::size_t>()) +
                                    " tokens, the dumps hold " + std::to_string(tokens));
    }
    if (entry.at("hidden_size").get<std::size_t>() != hidden_size) {
        throw std::invalid_argument("case '" + name + "': manifest hidden_size disagrees with the "
                                    "head config");
    }
    Record record;
    record.token_ids = read_ids(directory / entry.at("ids").get<std::string>(), tokens);
    for (const Json& question : entry.at("questions")) {
        QuestionSpec spec;
        spec.question_id = question.at("question_id").get<std::string>();
        spec.type = static_cast<QuestionType>(question.at("type").get<std::int32_t>());
        spec.question = span_of(question.at("question_span"), name, spec.question_id + " question span");
        for (const Json& option : question.at("option_spans")) {
            spec.options.push_back(
                span_of(option, name, spec.question_id + " option span"));
        }
        for (const Json& option_id : question.at("option_ids")) {
            spec.option_ids.push_back(option_id.get<std::string>());
        }
        record.questions.push_back(std::move(spec));
    }
    return record;
}

struct CaseResult {
    std::string name;
    std::size_t tokens = 0;
    std::size_t questions = 0;
    std::size_t options = 0;
    std::size_t argmax_agreed = 0;
    float max_abs_diff = 0.0F;
    std::string worst;
    std::string note;
};

// Both heads expose the same contract -- `score(hidden, record)` returns one logit vector per question
// -- so the comparison below is written once and the gold key says which head's expectations this case
// carries. A parity gate that reimplements the diff per head is a gate that drifts.
template <typename Head>
[[nodiscard]] CaseResult replay(const Head& head, const Json& entry,
                                const std::filesystem::path& directory, std::string_view gold_key) {
    const std::string name = entry.at("name").get<std::string>();
    const std::size_t hidden_size = static_cast<std::size_t>(head.config().hidden_size);
    const std::size_t rows = entry.at("tokens").get<std::size_t>();
    CaseResult result;
    result.name = name;
    result.tokens = rows;

    const std::vector<float> hidden =
        read_hidden(directory / entry.at("hidden").get<std::string>(), rows, hidden_size);
    const Record record = record_from_manifest(entry, directory, rows, hidden_size);
    const std::vector<std::vector<float>> logits = head.score(std::span<const float>(hidden), record);

    const Json& expected = entry.at(std::string(gold_key));
    if (logits.size() != expected.size() || logits.size() != record.questions.size()) {
        throw std::invalid_argument("case '" + name + "': the head answered " +
                                    std::to_string(logits.size()) + " questions for " +
                                    std::to_string(expected.size()) + " expected");
    }
    for (std::size_t index = 0; index < logits.size(); ++index) {
        const std::vector<float>& got = logits[index];
        const std::vector<float> want = expected[index].get<std::vector<float>>();
        result.questions += 1;
        result.options += got.size();
        if (got.size() != want.size()) {
            throw std::invalid_argument("case '" + name + "' question " +
                                        std::to_string(index) + ": " + std::to_string(got.size()) +
                                        " logits against " + std::to_string(want.size()));
        }
        std::size_t got_peak = 0;
        std::size_t want_peak = 0;
        for (std::size_t option = 0; option < got.size(); ++option) {
            const float delta = std::fabs(got[option] - want[option]);
            if (delta > result.max_abs_diff) {
                result.max_abs_diff = delta;
                result.worst = name + "/" + record.questions[index].question_id + " option " +
                               std::to_string(option) + " (" + std::to_string(got[option]) +
                               " against gold " + std::to_string(want[option]) + ")";
            }
            if (got[option] > got[got_peak]) {
                got_peak = option;
            }
            if (want[option] > want[want_peak]) {
                want_peak = option;
            }
        }
        if (got_peak == want_peak) {
            result.argmax_agreed += 1;
        } else {
            result.note += " argmax flip on " + record.questions[index].question_id;
        }
    }
    return result;
}

void usage() {
    std::printf(
        "usage: ninfer-decide-serve --selftest <decide_fixtures dir> [--max-abs-diff <float>]\n"
        "       ninfer-decide-serve --student-selftest <student_fixtures dir> "
        "[--max-abs-diff <float>]\n"
        "       ninfer-decide-serve --encode-check <gate_prompts.json> <release dir> "
        "<decide_fixtures dir>\n"
        "\n"
        "  --selftest <dir>   replay <dir>/manifest.json through the C++ head and compare each\n"
        "                     case against its expected_logits. CPU only: no engine, no GPU.\n"
        "  --student-selftest <dir>\n"
        "                     the same gate for the CLM-style student head: compare its logits against\n"
        "                     expected_student_logits, and when the manifest also carries the released\n"
        "                     head, time both over identical records.\n"
        "  --max-abs-diff f   fail the run if any logit differs from the gold by more than f\n"
        "                     (default 0 = report the numbers without gating on them)\n"
        "  --encode-check ... rebuild each record's token ids and question spans with the engine's BPE\n"
        "                     frontend and compare them against the stored dumps. Also CPU only.\n"
        "  --serve ...        load an artifact and answer POST /v1/decide from the C++ head. Run\n"
        "                     `ninfer-decide-serve --serve --help` for that mode's flags.\n");
}

int encode_check(const Options& options) {
    const std::filesystem::path release(options.release);
    const std::string tokenizer_json = read_text(release / "tokenizer.json");
    const std::string tokenizer_config_json = read_text(release / "tokenizer_config.json");
    const std::string generation_config_json = read_text(release / "generation_config.json");
    const ninfer::models::qwen3_5::frontend::TokenizerResources resources {
        tokenizer_json, tokenizer_config_json, generation_config_json};
    ninfer::models::qwen3_5::frontend::Tokenizer tokenizer(resources);
    const EngineFragmentEncoder encoder(tokenizer);

    const Request prompts = Request::parse(read_text(std::filesystem::path(options.prompts)));
    const Json manifest = Json::parse(read_text(std::filesystem::path(options.fixtures) /
                                                "manifest.json"));
    const std::filesystem::path directory(options.fixtures);
    if (!prompts.is_array()) {
        std::fprintf(stderr, "encode-check: %s is not a list of records\n", options.prompts.c_str());
        return 2;
    }

    std::size_t compared = 0;
    std::size_t mismatched = 0;
    for (std::size_t index = 0; index < prompts.size(); ++index) {
        const std::string case_name = "clef_p" + std::to_string(index);
        const Json* entry = nullptr;
        for (const Json& candidate : manifest.at("cases")) {
            if (candidate.at("name").get<std::string>() == case_name) {
                entry = &candidate;
                break;
            }
        }
        if (entry == nullptr) {
            std::fprintf(stderr, "encode-check: no fixture case named %s\n", case_name.c_str());
            return 2;
        }
        const std::size_t tokens = entry->at("tokens").get<std::size_t>();
        const std::vector<std::int32_t> stored =
            read_ids(directory / entry->at("ids").get<std::string>(), tokens);
        const Record got = encode_record(prompts.at(index), encoder);

        std::size_t first_difference = got.token_ids.size();
        for (std::size_t offset = 0; offset < std::min(got.token_ids.size(), stored.size());
             ++offset) {
            if (got.token_ids[offset] != stored[offset]) {
                first_difference = offset;
                break;
            }
        }
        bool ids_agree = first_difference == stored.size() && got.token_ids.size() == stored.size();

        std::size_t span_problems = 0;
        if (got.questions.size() != entry->at("questions").size()) {
            ++span_problems;
        }
        for (std::size_t question = 0;
             question < std::min(got.questions.size(), entry->at("questions").size()); ++question) {
            const QuestionSpec& spec = got.questions[question];
            const Json& expected = entry->at("questions")[question];
            if (spec.question_id != expected.at("question_id").get<std::string>()) {
                ++span_problems;
            }
            if (static_cast<std::int32_t>(spec.type) != expected.at("type").get<std::int32_t>()) {
                ++span_problems;
            }
            if (spec.question.begin != expected.at("question_span")[0].get<std::size_t>() ||
                spec.question.end != expected.at("question_span")[1].get<std::size_t>()) {
                ++span_problems;
            }
            if (spec.options.size() != expected.at("option_spans").size()) {
                ++span_problems;
            }
            for (std::size_t option = 0;
                 option < std::min(spec.options.size(), expected.at("option_spans").size());
                 ++option) {
                if (spec.options[option].begin !=
                    expected.at("option_spans")[option][0].get<std::size_t>() ||
                    spec.options[option].end !=
                        expected.at("option_spans")[option][1].get<std::size_t>()) {
                    ++span_problems;
                }
                if (spec.option_ids[option] !=
                    expected.at("option_ids")[option].get<std::string>()) {
                    ++span_problems;
                }
            }
        }
        ++compared;
        if (!ids_agree || span_problems != 0) {
            ++mismatched;
        }
        std::printf("%-10s ids %s (%zu got / %zu stored, first difference at %zu)  questions %zu  "
                    "span fields wrong %zu\n",
                    case_name.c_str(), ids_agree ? "AGREE" : "DIFFER", got.token_ids.size(),
                    stored.size(), first_difference, got.questions.size(), span_problems);
    }
    std::printf("\n%zu records, %zu mismatched against the stored dumps\n", compared, mismatched);
    return mismatched == 0 ? 0 : 1;
}

int selftest(const Options& options) {
    const std::filesystem::path directory(options.fixtures);
    std::FILE* handle = std::fopen((directory / "manifest.json").string().c_str(), "rb");
    if (handle == nullptr) {
        std::fprintf(stderr, "selftest: no manifest.json in %s\n", options.fixtures.c_str());
        return 2;
    }
    Json manifest;
    try {
        manifest = Json::parse(handle);
    } catch (const std::exception& error) {
        std::fclose(handle);
        std::fprintf(stderr, "selftest: manifest.json did not parse: %s\n", error.what());
        return 2;
    }
    std::fclose(handle);

    const Json& config_json = manifest.at("head_config");
    HeadConfig config;
    config.hidden_size = config_json.at("hidden_size").get<int>();
    config.width = config_json.at("width").get<int>();
    config.routing_layers = config_json.at("routing_layers").get<int>();
    config.layers = config_json.at("layers").get<int>();
    config.heads = config_json.at("heads").get<int>();
    config.feedforward = config_json.at("feedforward").get<int>();

    std::printf("head config: hidden %d width %d routing %d layers %d heads %d ff %d\n",
                config.hidden_size, config.width, config.routing_layers, config.layers, config.heads,
                config.feedforward);
    const auto opened = std::chrono::steady_clock::now();
    HeadWeights weights = HeadWeights::open(manifest.at("head_sidecar").get<std::string>());
    std::printf("head weights: %zu parameters, %.3f s\n", weights.parameter_count(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - opened).count());
    LexicalTable lexical = LexicalTable::open(manifest.at("lexical_sidecar").get<std::string>());
    std::printf("lexical table: %lld x %lld\n",
                static_cast<long long>(lexical.rows()), static_cast<long long>(lexical.columns()));

    JointSchemaHead head(config, weights, lexical);

    std::size_t argmax_total = 0;
    std::size_t argmax_agreed = 0;
    float worst = 0.0F;
    std::string worst_case;
    double score_seconds = 0.0;
    for (const Json& entry : manifest.at("cases")) {
        const auto started = std::chrono::steady_clock::now();
        const CaseResult result = replay(head, entry, directory, "expected_logits");
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        score_seconds += seconds;
        std::printf("%-12s %3zu tokens  %zu q/%zu opt  argmax %zu/%zu  max|diff| %.6e  %.2f s%s\n",
                    result.name.c_str(), result.tokens, result.questions, result.options,
                    result.argmax_agreed, result.questions, result.max_abs_diff, seconds,
                    result.note.c_str());
        argmax_total += result.questions;
        argmax_agreed += result.argmax_agreed;
        if (result.max_abs_diff > worst) {
            worst = result.max_abs_diff;
            worst_case = result.worst;
        }
    }
    std::printf("\n%d cases, %zu questions: argmax %zu/%zu, worst max|diff| %.6e at %s\n",
                static_cast<int>(manifest.at("cases").size()), argmax_total, argmax_agreed,
                argmax_total, worst, worst_case.empty() ? "none" : worst_case.c_str());
    std::printf("released head score time: %.3f s over %zu cases (%.1f ms per record)\n",
                score_seconds, manifest.at("cases").size(),
                1000.0 * score_seconds / static_cast<double>(manifest.at("cases").size()));
    std::printf("lexical rows touched: %zu\n", lexical.cached_rows());
    if (options.max_abs_diff > 0.0F && worst > options.max_abs_diff) {
        std::printf("GATED FAIL: %.6e exceeds the allowed %.6e\n", worst, options.max_abs_diff);
        return 1;
    }
    if (argmax_agreed != argmax_total) {
        return 1;
    }
    return 0;
}

// The student leg of the gate. Same shape as `selftest`, with two additions the decision-speed claim
// needs: the manifest may carry the released head's sidecar and gold alongside the student's, and both
// heads are then timed over the identical hidden/record set from their own fresh lexical cache -- the
// comparison is only worth printing if neither head borrows the other's warmed rows.
int student_selftest(const Options& options) {
    const std::filesystem::path directory(options.fixtures);
    const Json manifest = Json::parse(read_text(directory / "manifest.json"));

    const Json& config_json = manifest.at("student_config");
    StudentConfig config;
    config.hidden_size = config_json.at("hidden_size").get<std::size_t>();
    config.width = config_json.at("width").get<std::size_t>();
    config.types = config_json.at("types").get<std::size_t>();
    std::printf("student config: hidden %zu width %zu types %zu\n", config.hidden_size, config.width,
                config.types);

    const auto opened = std::chrono::steady_clock::now();
    HeadWeights student_weights = HeadWeights::open(manifest.at("student_sidecar").get<std::string>());
    const double student_load =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - opened).count();
    std::printf("student weights: %.1fM parameters, sidecar open %.3f s\n",
                static_cast<double>(student_weights.parameter_count()) / 1e6, student_load);
    LexicalTable student_lexical =
        LexicalTable::open(manifest.at("lexical_sidecar").get<std::string>());
    StudentHead head(config, student_weights, student_lexical);
    std::printf("student logit scale: %.4f\n", head.logit_scale());

    struct HeadTiming {
        std::size_t questions = 0;
        std::size_t agreed = 0;
        std::size_t total = 0;
        double seconds = 0.0;
        double first_seconds = 0.0;
        float worst = 0.0F;
        std::string worst_case;
    };

    auto sweep = [&directory](const auto& scored, const Json& entries, std::string_view gold_key,
                              const char* label, HeadTiming& timing) {
        for (std::size_t index = 0; index < entries.size(); ++index) {
            const auto started = std::chrono::steady_clock::now();
            const CaseResult result = replay(scored, entries[index], directory, gold_key);
            const double seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            timing.seconds += seconds;
            if (index == 0) {
                timing.first_seconds = seconds;
            }
            std::printf("%-14s %4zu tokens  %zu q/%zu opt  argmax %zu/%zu  max|diff| %.6e  %6.2f ms%s\n",
                        result.name.c_str(), result.tokens, result.questions, result.options,
                        result.argmax_agreed, result.questions, result.max_abs_diff, seconds * 1000.0,
                        result.note.c_str());
            timing.questions += result.questions;
            timing.agreed += result.argmax_agreed;
            timing.total += result.questions;
            if (result.max_abs_diff > timing.worst) {
                timing.worst = result.max_abs_diff;
                timing.worst_case = result.worst;
            }
        }
        std::printf("\n%s: %zu cases, %zu questions, argmax %zu/%zu, worst max|diff| %.6e at %s\n",
                    label, entries.size(), timing.total, timing.agreed, timing.total, timing.worst,
                    timing.worst_case.empty() ? "none" : timing.worst_case.c_str());
        std::printf("%s: mean %.1f ms per record (first %.1f ms, cold lexical cache)\n", label,
                    1000.0 * timing.seconds / static_cast<double>(entries.size()),
                    1000.0 * timing.first_seconds);
    };

    HeadTiming student;
    sweep(head, manifest.at("cases"), "expected_student_logits", "student", student);

    if (manifest.contains("released_sidecar")) {
        HeadWeights released_weights =
            HeadWeights::open(manifest.at("released_sidecar").get<std::string>());
        const Json& released_config_json = manifest.at("released_config");
        HeadConfig released_config;
        released_config.hidden_size = released_config_json.at("hidden_size").get<int>();
        released_config.width = released_config_json.at("width").get<int>();
        released_config.routing_layers = released_config_json.at("routing_layers").get<int>();
        released_config.layers = released_config_json.at("layers").get<int>();
        released_config.heads = released_config_json.at("heads").get<int>();
        released_config.feedforward = released_config_json.at("feedforward").get<int>();
        LexicalTable released_lexical =
            LexicalTable::open(manifest.at("lexical_sidecar").get<std::string>());
        JointSchemaHead released(released_config, released_weights, released_lexical);
        std::printf("\nreleased head: %.1fM parameters, width %d layers %d routing %d\n",
                    static_cast<double>(released_weights.parameter_count()) / 1e6,
                    released_config.width, released_config.layers, released_config.routing_layers);
        HeadTiming baseline;
        sweep(released, manifest.at("cases"), "expected_released_logits", "released head", baseline);
        if (student.seconds > 0.0 && baseline.questions == student.questions) {
            std::printf("\nscore cost: student %.1f ms/record vs released %.1f ms/record = %.1fx "
                        "cheaper, same %zu questions\n",
                        1000.0 * student.seconds / static_cast<double>(manifest.at("cases").size()),
                        1000.0 * baseline.seconds / static_cast<double>(manifest.at("cases").size()),
                        baseline.seconds / student.seconds, student.total);
        } else {
            std::printf("\nscore cost: student %.1f ms/record (no released baseline)\n",
                        1000.0 * student.seconds / static_cast<double>(manifest.at("cases").size()));
        }
    }

    std::printf("student lexical rows touched: %zu\n", student_lexical.cached_rows());
    if (student.agreed != student.total) {
        return 1;
    }
    if (options.max_abs_diff > 0.0F && student.worst > options.max_abs_diff) {
        std::printf("GATED FAIL: %.6e exceeds the allowed %.6e\n", student.worst,
                    options.max_abs_diff);
        return 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    // Unbuffered: the selftest writes its numbers to a log while the failure text goes to stderr, and
    // a block-buffered stdout reorders the two when the log is a file.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "--serve") == 0) {
            return ninfer::decide::serve_main(argc, argv);
        }
    }
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            usage();
            return 0;
        }
        if (argument == "--selftest" && index + 1 < argc) {
            options.mode = "selftest";
            options.fixtures = argv[++index];
            continue;
        }
        if (argument == "--student-selftest" && index + 1 < argc) {
            options.mode = "student-selftest";
            options.fixtures = argv[++index];
            continue;
        }
        if (argument == "--encode-check" && index + 3 < argc) {
            options.mode = "encode-check";
            options.prompts = argv[++index];
            options.release = argv[++index];
            options.fixtures = argv[++index];
            continue;
        }
        if (argument == "--max-abs-diff" && index + 1 < argc) {
            options.max_abs_diff = std::stof(argv[++index]);
            continue;
        }
        std::fprintf(stderr, "unknown argument '%s'\n\n", argument.c_str());
        usage();
        return 2;
    }
    if (options.mode == "encode-check") {
        try {
            return encode_check(options);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "encode-check failed: %s\n", error.what());
            return 3;
        }
    }
    if (options.mode == "student-selftest") {
        try {
            return student_selftest(options);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "student-selftest failed: %s\n", error.what());
            return 3;
        }
    }
    if (options.mode != "selftest") {
        std::fprintf(stderr,
                     "ninfer-decide-serve needs one mode: --selftest, --student-selftest, "
                     "--encode-check or --serve.\n");
        usage();
        return 2;
    }
    try {
        return selftest(options);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "selftest failed: %s\n", error.what());
        return 3;
    }
}
