// ninfer-decide-serve --serve: the decision head running inside the engine process.
//
// One artifact load, one HTTP front, and a record goes in and comes back as a decision. The path is
// `encode_record` (engine artifact tokenizer) -> `Engine::export_hidden` (prefill only) -> head ->
// softmax, all in this process: no Python head, no second round trip, no hidden state on the wire.
// That is what makes the latency number the decision path actually has, rather than the sum of numbers
// measured in three different programs.
//
//   POST /v1/decide   {"head":"student", ...the Jev/SystemOne record body...}  -> decisions + timings
//   GET  /health
//
// Two heads can be loaded at once (`--student-sidecar` and `--joint-sidecar`); the request body picks
// which one answers, so a capability or cost difference between them is measured against the same
// tower, the same tokenizer and the same lexical table instead of across two engine startups.
//
// The engine owns one execution slot and `LexicalTable` mutates its row cache under a non-const
// reference, so every handler takes the same mutex. Concurrency here would be a crash, not a feature.

#include "joint_head/bf16.h"
#include "joint_head/encoder.h"
#include "joint_head/head.h"
#include "joint_head/record.h"
#include "joint_head/student.h"
#include "joint_head/weights.h"

#include "ninfer/engine.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::decide {
namespace {

using Json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;

struct Options {
    std::string artifact;
    std::string host = "127.0.0.1";
    int port = 8198;
    int device = 0;
    std::uint32_t max_context = 8192;
    std::uint32_t prefill_chunk = 1024;
    ninfer::KvCacheStorage kv = ninfer::KvCacheStorage::Nvfp4Group16;
    std::string lexical_sidecar;
    std::string student_sidecar;
    std::string joint_sidecar;
    std::string release_dir;  // for joint_head_config.json, the released head's shape file
    std::string api_key;
};

[[noreturn]] void fail(const std::string& message) {
    std::cerr << "decide-serve: " << message << '\n';
    std::exit(2);
}

void usage(std::string_view argv0) {
    std::cout << "usage: " << argv0 << " --serve --artifact PATH --lexical-sidecar PATH"
              << " [--student-sidecar PATH] [--joint-sidecar PATH --release DIR] [options]\n"
              << "  --port N             listen port (default 8198)\n"
              << "  --max-context N      token ceiling per record (default 8192)\n"
              << "  --prefill-chunk N    tokens per forward step, multiple of 128 (default 1024)\n"
              << "  --kv-dtype D         bf16|int8|fp8|nvfp4|k8v4 (default nvfp4)\n"
              << "  --api-key KEY        require Authorization: Bearer KEY\n";
}

Json parse_body(const std::string& text) {
    const Json parsed = Json::parse(text, nullptr, false);
    if (parsed.is_discarded()) {
        throw std::invalid_argument("request body is not JSON");
    }
    return parsed;
}

void reply_error(httplib::Response& response, int code, const std::string& message) {
    Json body;
    body["error"] = message;
    response.status = code;
    response.set_content(body.dump(), "application/json");
}

bool authorized(const httplib::Request& request, const std::string& api_key) {
    if (api_key.empty()) {
        return true;
    }
    return request.get_header_value("Authorization") == "Bearer " + api_key;
}

// The engine's artifact tokenizer, handed to the head encoder through the one method it needs. The
// header for `tokenize_text` is explicit that no chat template and no implicit special token is added,
// which is the contract `encode_record`'s fragment-by-fragment BPE runs depend on.
class EngineFragmentEncoder final : public joint_head::FragmentEncoder {
public:
    explicit EngineFragmentEncoder(const ninfer::Engine& engine) : engine_(engine) {}

    [[nodiscard]] std::vector<std::int32_t> encode(std::string_view fragment) const override {
        return engine_.tokenize_text(fragment);
    }

private:
    const ninfer::Engine& engine_;
};

[[nodiscard]] std::vector<float> softmax(std::vector<float> logits) {
    const float peak = *std::max_element(logits.begin(), logits.end());
    double total = 0.0;
    for (float& value : logits) {
        value = std::exp(value - peak);
        total += value;
    }
    const float inverse = static_cast<float>(1.0 / total);
    for (float& value : logits) {
        value *= inverse;
    }
    return logits;
}

// The student sidecar is self-describing: `state.0.weight` is [width, 2 * hidden_size] and the type
// embedding's row count is the number of question types. Reading the shape back beats a second config
// file that could disagree with the weights it is describing.
[[nodiscard]] joint_head::StudentConfig student_config_from(const joint_head::HeadWeights& weights) {
    const auto& state_in = weights.tensor("state.0.weight").dims;
    joint_head::StudentConfig config;
    config.width = static_cast<std::size_t>(state_in[0]);
    config.hidden_size = static_cast<std::size_t>(state_in[1]) / 2U;
    config.types = static_cast<std::size_t>(weights.tensor("type_embedding.weight").dims[0]);
    return config;
}

[[nodiscard]] joint_head::HeadConfig joint_config_from(const std::string& release_dir) {
    const std::filesystem::path path(release_dir);
    std::ifstream input(path / "joint_head_config.json");
    if (!input) {
        throw std::invalid_argument("decide-serve: cannot read " +
                                    (path / "joint_head_config.json").string());
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    const Json parsed = parse_body(buffer.str());
    joint_head::HeadConfig config;
    config.hidden_size = parsed.at("hidden_size").get<int>();
    config.width = parsed.at("width").get<int>();
    config.routing_layers = parsed.at("routing_layers").get<int>();
    config.layers = parsed.at("layers").get<int>();
    config.heads = parsed.at("heads").get<int>();
    config.feedforward = parsed.at("feedforward").get<int>();
    return config;
}

struct Server {
    ninfer::Engine engine;
    std::mutex gate;
    std::unique_ptr<joint_head::LexicalTable> lexical;
    std::unique_ptr<joint_head::StudentHead> student;
    std::unique_ptr<joint_head::JointSchemaHead> joint;
    std::string model_name;
    std::string default_head;
    std::atomic<std::uint64_t> decisions {0};
    EngineFragmentEncoder encoder;

    Server(Options options, ninfer::Engine loaded, std::string name)
        : engine(std::move(loaded)),
          lexical(std::make_unique<joint_head::LexicalTable>(
              joint_head::LexicalTable::open(options.lexical_sidecar))),
          model_name(std::move(name)),
          encoder(engine) {
        if (!options.student_sidecar.empty()) {
            auto weights = std::make_unique<joint_head::HeadWeights>(
                joint_head::HeadWeights::open(options.student_sidecar));
            // The config has to outlive the pointer, so it is derived and stored before construction.
            student_config_ = student_config_from(*weights);
            joint_head::validate_student_shapes(*weights, student_config_);
            student = std::make_unique<joint_head::StudentHead>(student_config_, *weights, *lexical);
            student_weights_ = std::move(weights);
            default_head = "student";
        }
        if (!options.joint_sidecar.empty()) {
            joint_weights_ = std::make_unique<joint_head::HeadWeights>(
                joint_head::HeadWeights::open(options.joint_sidecar));
            joint_config_ = joint_config_from(options.release_dir);
            joint_head::validate_head_shapes(*joint_weights_, joint_config_.hidden_size,
                                             joint_config_.width, joint_config_.heads,
                                             joint_config_.feedforward, joint_config_.routing_layers,
                                             joint_config_.layers);
            joint = std::make_unique<joint_head::JointSchemaHead>(joint_config_, *joint_weights_,
                                                                  *lexical);
            if (default_head.empty()) {
                default_head = "joint";
            }
        }
    }

    joint_head::StudentConfig student_config_;
    joint_head::HeadConfig joint_config_;
    std::unique_ptr<joint_head::HeadWeights> student_weights_;
    std::unique_ptr<joint_head::HeadWeights> joint_weights_;

    [[nodiscard]] Json decide(const Json& body, std::uint32_t max_context) {
        const std::string which = body.contains("head") ? body.at("head").get<std::string>()
                                                        : default_head;
        joint_head::Record record;
        double encode_ms = 0.0;
        double prefill_ms = 0.0;
        double head_ms = 0.0;
        std::vector<std::vector<float>> logits;
        std::size_t hidden_size = 0;
        {
            std::scoped_lock lock(gate);
            {
                const auto started = Clock::now();
                joint_head::EncodeRecordOptions encode_options;
                encode_options.max_length = max_context;
                record = joint_head::encode_record(body, encoder, encode_options);
                encode_ms = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
            }
            {
                const auto started = Clock::now();
                current_hidden_ = engine.export_hidden(std::vector<ninfer::TokenId>(
                    record.token_ids.begin(), record.token_ids.end()));
                prefill_ms = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
                hidden_size = current_hidden_.hidden_size;
            }
            {
                const auto started = Clock::now();
                const std::vector<float> hidden =
                    joint_head::widen(std::span<const std::uint16_t>(current_hidden_.bf16));
                if (which == "student") {
                    if (!student) {
                        throw std::invalid_argument("no --student-sidecar was loaded");
                    }
                    logits = student->score(std::span<const float>(hidden), record);
                } else if (which == "joint") {
                    if (!joint) {
                        throw std::invalid_argument("no --joint-sidecar was loaded");
                    }
                    logits = joint->score(std::span<const float>(hidden), record);
                } else {
                    throw std::invalid_argument("head must be 'student' or 'joint'");
                }
                head_ms = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
            }
        }
        current_hidden_.bf16.clear();
        current_hidden_.bf16.shrink_to_fit();

        Json questions = Json::array();
        for (std::size_t index = 0; index < record.questions.size(); ++index) {
            const joint_head::QuestionSpec& question = record.questions[index];
            Json probabilities = Json::array();
            for (const float value : softmax(logits.at(index))) {
                probabilities.push_back(value);
            }
            std::size_t peak = 0;
            for (std::size_t option = 1; option < logits.at(index).size(); ++option) {
                if (probabilities[option].get<float>() > probabilities[peak].get<float>()) {
                    peak = option;
                }
            }
            Json logits_json = Json::array();
            for (const float value : logits.at(index)) {
                logits_json.push_back(value);
            }
            questions.push_back(Json {{"question_id", question.question_id},
                                      {"option_ids", question.option_ids},
                                      {"answer", question.option_ids.at(peak)},
                                      {"probability", probabilities[peak]},
                                      {"probabilities", probabilities},
                                      {"logits", logits_json}});
        }
        return Json {{"model", model_name},
                     {"head", which},
                     {"tokens", record.tokens()},
                     {"hidden_size", hidden_size},
                     {"questions", questions},
                     {"timings_ms", {{"encode", encode_ms},
                                     {"prefill", prefill_ms},
                                     {"head", head_ms},
                                     {"total", encode_ms + prefill_ms + head_ms}}}};
    }

private:
    ninfer::HiddenExport current_hidden_;
};

int serve(const Options& options) {
    ninfer::EngineOptions engine_options;
    engine_options.artifact_path = options.artifact;
    engine_options.purpose = ninfer::EnginePurpose::CausalScoring;
    engine_options.device = options.device;
    engine_options.max_context = options.max_context;
    engine_options.prefill_chunk = options.prefill_chunk;
    engine_options.kv_cache = options.kv;

    std::cout << "[decide-serve] loading artifact " << options.artifact << " ...\n" << std::flush;
    const auto loaded_at = Clock::now();
    ninfer::Engine engine(std::move(engine_options));
    const double load_seconds = std::chrono::duration<double>(Clock::now() - loaded_at).count();
    const ninfer::LoadSummary load = engine.load_summary();
    std::cout << "[decide-serve] engine ready model=" << load.model_name << " max_context="
              << options.max_context << " in " << load_seconds << "s\n"
              << std::flush;

    Server server(options, std::move(engine), load.model_name);
    std::cout << "[decide-serve] heads loaded: student=" << (server.student ? "yes" : "no")
              << " joint=" << (server.joint ? "yes" : "no")
              << " default=" << server.default_head << "\n"
              << std::flush;

    httplib::Server http;
    // A decision sends nothing back while the prefill steps run, so the 5 s defaults can close the
    // connection mid-work with nothing in the log (same mitigation as hidden-serve / ninfer-serve).
    http.set_keep_alive_timeout(120);
    http.set_read_timeout(300, 0);
    http.set_write_timeout(300, 0);
    http.set_payload_max_length(64ULL << 20);

    http.Get("/health", [&](const httplib::Request& request, httplib::Response& response) {
        if (!authorized(request, options.api_key)) {
            return reply_error(response, 401, "invalid api key");
        }
        response.set_content(
            Json {{"status", "ok"},
                  {"model", server.model_name},
                  {"max_context", options.max_context},
                  {"heads", Json {{"student", static_cast<bool>(server.student)},
                                  {"joint", static_cast<bool>(server.joint)}}},
                  {"decisions", server.decisions.load()}}
                .dump(),
            "application/json");
    });

    http.Post("/v1/decide", [&](const httplib::Request& request, httplib::Response& response) {
        if (!authorized(request, options.api_key)) {
            return reply_error(response, 401, "invalid api key");
        }
        try {
            const auto started = Clock::now();
            const Json body = parse_body(request.body);
            const Json answer = server.decide(body, options.max_context);
            const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
            server.decisions.fetch_add(1);
            std::cout << "[decide-serve] head=" << answer.at("head") << " tokens="
                      << answer.at("tokens") << " questions=" << answer.at("questions").size()
                      << " prefill=" << answer.at("timings_ms").at("prefill")
                      << "ms head=" << answer.at("timings_ms").at("head") << "ms total=" << seconds
                      << "s\n"
                      << std::flush;
            response.set_content(answer.dump(), "application/json");
        } catch (const std::exception& error) {
            reply_error(response, 500, error.what());
        }
    });

    std::cout << "[decide-serve] listening on http://" << options.host << ":" << options.port
              << "\n"
              << std::flush;
    if (!http.listen(options.host, options.port)) {
        fail("could not bind " + options.host + ":" + std::to_string(options.port));
    }
    return 0;
}

ninfer::KvCacheStorage kv_from(const std::string& name) {
    if (name == "bf16") {
        return ninfer::KvCacheStorage::BFloat16;
    }
    if (name == "int8") {
        return ninfer::KvCacheStorage::Int8Group64;
    }
    if (name == "fp8") {
        return ninfer::KvCacheStorage::Fp8E4M3Row256;
    }
    if (name == "k8v4") {
        return ninfer::KvCacheStorage::Fp8KeyNvfp4Value;
    }
    if (name == "nvfp4") {
        return ninfer::KvCacheStorage::Nvfp4Group16;
    }
    fail("unknown --kv-dtype '" + name + "'");
}

} // namespace

// Entry point used by main() once it sees --serve. Argument parsing is deliberately a copy of
// hidden-serve's shape so the flags a user already knows mean the same thing in both programs.
int serve_main(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string flag = argv[index];
        auto take = [&](std::string_view name) -> std::string {
            if (index + 1 >= argc) {
                fail(std::string(name) + " needs a value");
            }
            return argv[++index];
        };
        if (flag == "--artifact") {
            options.artifact = take("--artifact");
        } else if (flag == "--host") {
            options.host = take("--host");
        } else if (flag == "--port") {
            options.port = std::stoi(take("--port"));
        } else if (flag == "--device") {
            options.device = std::stoi(take("--device"));
        } else if (flag == "--max-context") {
            options.max_context = static_cast<std::uint32_t>(std::stoul(take("--max-context")));
        } else if (flag == "--prefill-chunk") {
            options.prefill_chunk = static_cast<std::uint32_t>(std::stoul(take("--prefill-chunk")));
            // One exported forward step covers `prefill_chunk` tokens, and the engine rejects anything
            // that is not a multiple of its tile size -- fail here with the flag name instead.
            if (options.prefill_chunk % 128U != 0U) {
                fail("--prefill-chunk must be a multiple of 128");
            }
        } else if (flag == "--kv-dtype") {
            options.kv = kv_from(take("--kv-dtype"));
        } else if (flag == "--lexical-sidecar") {
            options.lexical_sidecar = take("--lexical-sidecar");
        } else if (flag == "--student-sidecar") {
            options.student_sidecar = take("--student-sidecar");
        } else if (flag == "--joint-sidecar") {
            options.joint_sidecar = take("--joint-sidecar");
        } else if (flag == "--release") {
            options.release_dir = take("--release");
        } else if (flag == "--api-key") {
            options.api_key = take("--api-key");
        } else if (flag == "--serve") {
            continue;
        } else if (flag == "--help" || flag == "-h") {
            usage("ninfer-decide-serve");
            return 0;
        } else {
            fail("unknown argument '" + flag + "'");
        }
    }
    if (options.artifact.empty() || options.lexical_sidecar.empty()) {
        usage("ninfer-decide-serve");
        fail("--serve requires --artifact and --lexical-sidecar");
    }
    if (options.student_sidecar.empty() && options.joint_sidecar.empty()) {
        fail("--serve requires at least one head: --student-sidecar or --joint-sidecar");
    }
    if (!options.joint_sidecar.empty() && options.release_dir.empty()) {
        fail("--joint-sidecar needs --release DIR for joint_head_config.json");
    }
    return serve(options);
}

} // namespace ninfer::decide
