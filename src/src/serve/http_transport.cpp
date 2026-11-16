#include "serve/http_transport.h"

#include "serve/request_validation.h"

#if defined(__linux__)
#    include <netinet/tcp.h>
#    include <sys/socket.h>
#elif defined(_WIN32)
#    include <mstcpip.h>
#endif

#include <stdexcept>
#include <utility>

namespace ninfer::serve {
namespace {

// Ported from shensanshu/ninfer-master-shensanshu-kvmem (Apache-2.0), 复现白皮书 §7.2. A non-streaming
// reply emits no bytes while prefill runs, so the library's 5 s defaults can close the connection while
// the engine is still working and logging nothing. This is a mitigation, not a fix: their root cause is
// unlocated, and a cold prefill beyond ~2 minutes can still be cut at the connection layer.
constexpr int kHttpKeepAliveWindowSeconds  = 120;
constexpr int kHttpReadWriteTimeoutSeconds = 300;

// Their 10 s / 3 s probe timing is relaabelled to 60 s / 15 s on this box: measured with SIO_KEEPALIVE_VALS
// at 10/3, a client that stops reading lets us queue one 64 KiB send buffer, and the probe then aborts the
// connection locally at ~116.7 s and discards those bytes (engine log: "cancelled during transport", HTTP
// 499; client received 0 B). At 60/15 the same stall is ruled by the 300 s write timer instead and the
// whole queue is delivered. All five connection judgements read identically for 60/15 and for "no option
// at all", so this keeps their fix and only retimes it.
constexpr int kKeepAliveIdleSeconds     = 60;
constexpr int kKeepAliveIntervalSeconds = 15;

#if defined(__linux__)
constexpr int kKeepAliveProbeCount                 = 3;
constexpr unsigned int kTcpUserTimeoutMilliseconds = 15000;

template <class T>
void set_socket_option(socket_t socket, int level, int option, const T& value) noexcept {
    (void)::setsockopt(socket, level, option, &value, sizeof(value));
}
#endif

} // namespace

RequestJson parse_json_body(const httplib::Request& request) {
    try {
        return RequestJson::parse(request.body);
    } catch (const std::exception&) { bad_request("request body is not valid JSON"); }
}

bool client_disconnected(const httplib::Request& request) { return request.is_connection_closed(); }

void prepare_sse_response(httplib::Response& response) {
    response.set_header("Cache-Control", "no-cache");
    response.set_header("X-Accel-Buffering", "no");
}

SseTransport::SseTransport(httplib::DataSink& sink, std::atomic<bool>& cancelled,
                           Clock::duration heartbeat_interval, Clock::time_point now)
    : sink_(sink), cancelled_(cancelled), heartbeat_interval_(heartbeat_interval),
      last_write_(now) {
    if (heartbeat_interval_ <= Clock::duration::zero()) {
        throw std::invalid_argument("SSE heartbeat interval must be positive");
    }
}

bool SseTransport::mark_cancelled() noexcept {
    cancelled_.store(true, std::memory_order_release);
    return true;
}

void SseTransport::write(std::string_view item, Clock::time_point now) {
    if (cancelled_.load(std::memory_order_acquire) || !sink_.write(item.data(), item.size())) {
        mark_cancelled();
        throw ClientDisconnected();
    }
    last_write_ = now;
}

void SseTransport::write(const std::vector<std::string>& items, Clock::time_point now) {
    for (const std::string& item : items) { write(item, now); }
}

bool SseTransport::poll(Clock::time_point now) {
    if (cancelled_.load(std::memory_order_acquire)) { return true; }
    if (sink_.is_writable && !sink_.is_writable()) { return mark_cancelled(); }
    if (now - last_write_ < heartbeat_interval_) { return false; }
    if (!sink_.write(kHeartbeatComment.data(), kHeartbeatComment.size())) {
        return mark_cancelled();
    }
    last_write_ = now;
    return false;
}

void configure_http_server_socket(socket_t socket) noexcept {
    httplib::default_socket_options(socket);
#if defined(__linux__)
    const int enabled = 1;
    set_socket_option(socket, SOL_SOCKET, SO_KEEPALIVE, enabled);
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPIDLE, kKeepAliveIdleSeconds);
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPINTVL, kKeepAliveIntervalSeconds);
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPCNT, kKeepAliveProbeCount);
    set_socket_option(socket, IPPROTO_TCP, TCP_USER_TIMEOUT, kTcpUserTimeoutMilliseconds);
#elif defined(_WIN32)
    const BOOL enabled = TRUE;
    (void)::setsockopt(socket, SOL_SOCKET, SO_KEEPALIVE, reinterpret_cast<const char*>(&enabled),
                       sizeof(enabled));
    // Windows has no TCP_KEEPIDLE/TCP_KEEPINTVL: the same 10 s idle / 3 s interval probe timing is set
    // per socket with SIO_KEEPALIVE_VALS.
    tcp_keepalive probes{};
    probes.onoff              = 1UL;
    probes.keepalivetime      = static_cast<unsigned long>(kKeepAliveIdleSeconds) * 1000UL;
    probes.keepaliveinterval  = static_cast<unsigned long>(kKeepAliveIntervalSeconds) * 1000UL;
    unsigned long bytes_returned = 0UL;
    (void)::WSAIoctl(socket, SIO_KEEPALIVE_VALS, &probes, sizeof(probes), nullptr, 0UL,
                     &bytes_returned, nullptr, nullptr);
#endif
}

void configure_http_server_timeouts(httplib::Server& server) noexcept {
    server.set_keep_alive_timeout(kHttpKeepAliveWindowSeconds);
    server.set_read_timeout(kHttpReadWriteTimeoutSeconds, 0);
    server.set_write_timeout(kHttpReadWriteTimeoutSeconds, 0);
}

void set_owned_json_content(httplib::Response& response, std::string body,
                            std::shared_ptr<RequestLifetime> lifetime) {
    response.set_content(std::move(body), "application/json");
    response.user_data.set("ninfer.request_lifetime", std::move(lifetime));
}

} // namespace ninfer::serve
