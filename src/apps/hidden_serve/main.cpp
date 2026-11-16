// ninfer-hidden-serve: prefill-only hidden-state export over HTTP.
//
// Loads an artifact with the CausalScoring engine purpose (single sequence, text backbone only, no
// speculative backend, no CUDA graphs) and answers POST /v1/hidden with the post-final-norm hidden
// state of every input token. Nothing is generated: the request carries token ids, the response is
// raw bfloat16, token-major, hidden_size values per token. Shape travels in response headers.
//
//   POST /v1/hidden    {"token_ids":[...]}    -> application/octet-stream
//   POST /v1/tokenize  {"text":"..."}         -> {"token_ids":[...]}
//   GET  /health                               -> {"status":"ok"}

#include "ninfer/engine.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

using Json      = nlohmann::json;
using Clock     = std::chrono::steady_clock;
using namespace std::string_literals;

namespace {

struct Options {
    std::string artifact;
    std::string host       = "127.0.0.1";
    int port               = 8197;
    int device             = 0;
    std::uint32_t max_context = 16384;
    std::uint32_t prefill_chunk = 1024;
    ninfer::KvCacheStorage kv  = ninfer::KvCacheStorage::Nvfp4Group16;
    std::string api_key;
    bool help = false;
};

[[noreturn]] void fail(const std::string& message) {
    std::cerr << "ninfer-hidden-serve: " << message << '\n';
    std::exit(2);
}

void usage(std::string_view argv0) {
    std::cout << "usage: " << argv0 << " --artifact PATH [options]\n"
              << "  --artifact PATH      .ninfer v3 artifact (required)\n"
              << "  --host HOST          bind address (default 127.0.0.1)\n"
              << "  --port N             listen port (default 8197)\n"
              << "  --device N           CUDA device index (default 0)\n"
              << "  --max-context N      token ceiling per request (default 16384)\n"
              << "  --prefill-chunk N    tokens per exported forward step, multiple of 128 (default 1024)\n"
              << "  --kv-dtype D         bf16|int8|fp8|nvfp4|k8v4 (default nvfp4)\n"
              << "  --api-key KEY        require Authorization: Bearer KEY\n";
}

std::uint32_t parse_positive(const char* text, std::string_view label) {
    char* end  = nullptr;
    const auto value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value == 0 || value > 0x7fffffffUL) {
        fail("invalid "s + std::string(label) + ": " + text);
    }
    return static_cast<std::uint32_t>(value);
}

ninfer::KvCacheStorage parse_kv(std::string_view value) {
    if (value == "bf16") { return ninfer::KvCacheStorage::BFloat16; }
    if (value == "int8") { return ninfer::KvCacheStorage::Int8Group64; }
    if (value == "fp8") { return ninfer::KvCacheStorage::Fp8E4M3Row256; }
    if (value == "nvfp4") { return ninfer::KvCacheStorage::Nvfp4Group16; }
    if (value == "k8v4") { return ninfer::KvCacheStorage::Fp8KeyNvfp4Value; }
    fail("invalid kv-dtype: "s + std::string(value));
}

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag(argv[i]);
        auto take = [&](std::string_view label) -> const char* {
            if (i + 1 >= argc) { fail(std::string(label) + " requires a value"); }
            return argv[++i];
        };
        if (flag == "--artifact") {
            options.artifact = take("--artifact");
        } else if (flag == "--host") {
            options.host = take("--host");
        } else if (flag == "--port") {
            options.port = static_cast<int>(parse_positive(take("--port"), "port"));
        } else if (flag == "--device") {
            options.device = static_cast<int>(parse_positive(take("--device"), "device"));
        } else if (flag == "--max-context") {
            options.max_context = parse_positive(take("--max-context"), "max-context");
        } else if (flag == "--prefill-chunk") {
            // One exported forward step covers `prefill_chunk` tokens, so this is both the prefill
            // batch shape and the pinned staging size (chunk x hidden x 2 bytes).
            options.prefill_chunk = parse_positive(take("--prefill-chunk"), "prefill-chunk");
            if (options.prefill_chunk % 128U != 0U) {
                fail("--prefill-chunk must be a multiple of 128");
            }
        } else if (flag == "--kv-dtype") {
            options.kv = parse_kv(take("--kv-dtype"));
        } else if (flag == "--api-key") {
            options.api_key = take("--api-key");
        } else if (flag == "--help" || flag == "-h") {
            options.help = true;
        } else {
            fail("unknown argument: "s + std::string(flag));
        }
    }
    return options;
}

bool authorized(const httplib::Request& request, const std::string& api_key) {
    if (api_key.empty()) { return true; }
    const std::string expected = "Bearer " + api_key;
    const auto given = request.get_header_value("Authorization");
    return given == expected;
}

void reply_error(httplib::Response& response, int status, const std::string& message) {
    Json body{{"error", {{"message", message}}}};
    response.status = status;
    response.set_content(body.dump(), "application/json");
}

std::vector<ninfer::TokenId> parse_token_ids(const std::string& body) {
    const Json parsed = Json::parse(body, nullptr, false);
    if (parsed.is_discarded()) { throw std::invalid_argument("request body is not valid JSON"); }
    if (!parsed.is_object() || !parsed.contains("token_ids") || !parsed["token_ids"].is_array()) {
        throw std::invalid_argument("request body requires a token_ids array");
    }
    std::vector<ninfer::TokenId> tokens;
    tokens.reserve(parsed["token_ids"].size());
    for (const auto& value : parsed["token_ids"]) {
        if (!value.is_number_integer()) {
            throw std::invalid_argument("token_ids must contain integers only");
        }
        tokens.push_back(value.get<ninfer::TokenId>());
    }
    return tokens;
}

} // namespace

int main(int argc, char** argv) {
    const Options options = parse(argc, argv);
    if (options.help || options.artifact.empty()) {
        usage(argv[0]);
        return options.help ? 0 : 2;
    }

    ninfer::EngineOptions engine_options;
    engine_options.artifact_path = options.artifact;
    engine_options.purpose       = ninfer::EnginePurpose::CausalScoring;
    engine_options.device        = options.device;
    engine_options.max_context   = options.max_context;
    engine_options.prefill_chunk = options.prefill_chunk;
    engine_options.kv_cache      = options.kv;

    std::cout << "[hidden-serve] loading artifact " << options.artifact << " ...\n" << std::flush;
    const auto loaded_at = Clock::now();
    ninfer::Engine engine(std::move(engine_options));
    const auto load_seconds =
        std::chrono::duration<double>(Clock::now() - loaded_at).count();
    const ninfer::LoadSummary load = engine.load_summary();
    std::cout << "[hidden-serve] engine ready model=" << load.model_name
              << " max_context=" << options.max_context << " in " << load_seconds << "s\n"
              << std::flush;

    // The scoring core owns one execution slot; serialize HTTP handlers instead of letting a second
    // concurrent request throw.
    std::mutex engine_mutex;
    std::atomic<std::uint64_t> exports{0};

    httplib::Server server;
    // A long export sends no bytes while the prefill steps run, so the library's 5 s defaults can close
    // the connection mid-work while the engine logs nothing (same mitigation as ninfer-serve).
    server.set_keep_alive_timeout(120);
    server.set_read_timeout(300, 0);
    server.set_write_timeout(300, 0);
    server.set_payload_max_length(64ULL << 20);

    server.Get("/health", [&](const httplib::Request& request, httplib::Response& response) {
        if (!authorized(request, options.api_key)) {
            return reply_error(response, 401, "invalid api key");
        }
        Json body{{"status", "ok"},
                  {"model", load.model_name},
                  {"max_context", options.max_context},
                  {"exports", exports.load()}};
        response.set_content(body.dump(), "application/json");
    });

    server.Post("/v1/tokenize", [&](const httplib::Request& request, httplib::Response& response) {
        if (!authorized(request, options.api_key)) {
            return reply_error(response, 401, "invalid api key");
        }
        try {
            const Json parsed = Json::parse(request.body, nullptr, false);
            if (parsed.is_discarded() || !parsed.is_object() || !parsed.contains("text") ||
                !parsed["text"].is_string()) {
                return reply_error(response, 400, "body requires a text string");
            }
            const std::string text = parsed["text"].get<std::string>();
            std::scoped_lock lock(engine_mutex);
            const auto tokens = engine.tokenize_text(text);
            Json body{{"token_ids", tokens}};
            response.set_content(body.dump(), "application/json");
        } catch (const std::exception& error) {
            reply_error(response, 500, error.what());
        }
    });

    server.Post("/v1/hidden", [&](const httplib::Request& request, httplib::Response& response) {
        if (!authorized(request, options.api_key)) {
            return reply_error(response, 401, "invalid api key");
        }
        try {
            std::vector<ninfer::TokenId> tokens = parse_token_ids(request.body);
            if (tokens.empty()) { return reply_error(response, 400, "token_ids is empty"); }

            const auto started = Clock::now();
            ninfer::HiddenExport hidden;
            {
                std::scoped_lock lock(engine_mutex);
                hidden = engine.export_hidden(std::move(tokens));
            }
            const auto seconds = std::chrono::duration<double>(Clock::now() - started).count();
            exports.fetch_add(1);

            response.set_header("X-Ninfer-Hidden-Tokens", std::to_string(hidden.token_count));
            response.set_header("X-Ninfer-Hidden-Size", std::to_string(hidden.hidden_size));
            response.set_header("X-Ninfer-Hidden-Dtype", "bf16");
            response.set_header("X-Ninfer-Elapsed-Ms", std::to_string(seconds * 1000.0));
            response.set_content(reinterpret_cast<const char*>(hidden.bf16.data()),
                                 hidden.size_bytes(), "application/octet-stream");
            std::cout << "[hidden-serve] export tokens=" << hidden.token_count
                      << " hidden=" << hidden.hidden_size << " in " << seconds << "s\n"
                      << std::flush;
        } catch (const std::exception& error) {
            reply_error(response, 500, error.what());
        }
    });

    std::cout << "[hidden-serve] listening on http://" << options.host << ":" << options.port
              << "\n"
              << std::flush;
    if (!server.listen(options.host, options.port)) {
        fail("could not bind "s + options.host + ":" + std::to_string(options.port));
    }
    return 0;
}
