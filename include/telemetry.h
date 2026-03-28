#pragma once

#include <cstdint>
#include <string>
#include <sstream>
#include "json.hpp"

using json = nlohmann::json;

// Telemetry output destinations (can be combined with bitwise OR)
enum class TelemetryOutput : uint8_t {
    None     = 0,
    Stdout   = 1 << 0,   // Log to console
    Response = 1 << 1,   // Include in HTTP response JSON
    // Future: File = 1 << 2, Prometheus = 1 << 3, etc.
};

// Bitwise operators for combining outputs
inline TelemetryOutput operator|(TelemetryOutput a, TelemetryOutput b) {
    return static_cast<TelemetryOutput>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}

inline bool has_telemetry_output(TelemetryOutput config, TelemetryOutput flag) {
    return (static_cast<uint8_t>(config) & static_cast<uint8_t>(flag)) != 0;
}

// Inference timing and token metrics (ticket-005)
// Only populated when telemetry is enabled.
struct InferenceMetrics {
    // Timing (milliseconds)
    int64_t preprocess_ms = 0;
    int64_t vision_encoder_ms = 0;
    int64_t prefill_ms = 0;
    int64_t decode_ms = 0;
    int64_t total_ms = 0;           // End-to-end latency

    // Token counts
    int input_tokens = 0;
    int output_tokens = 0;

    /**
     * Calculate decode speed (tokens/sec).
     * Returns 0.0 if decode_ms is 0.
     */
    float decode_tokens_per_sec() const {
        return decode_ms > 0 ? (output_tokens * 1000.0f / decode_ms) : 0.0f;
    }

    /**
     * Serialize to JSON for response output.
     */
    json to_json() const {
        return json{
            {"preprocess_ms", preprocess_ms},
            {"vision_encoder_ms", vision_encoder_ms},
            {"prefill_ms", prefill_ms},
            {"decode_ms", decode_ms},
            {"total_ms", total_ms},
            {"input_tokens", input_tokens},
            {"output_tokens", output_tokens},
            {"decode_tokens_per_sec", decode_tokens_per_sec()}
        };
    }
};

// Parse telemetry query parameter
// Examples: "" -> None, "1" -> Stdout, "stdout" -> Stdout, "response" -> Response, "stdout,response" -> Stdout|Response
inline TelemetryOutput parse_telemetry_param(const std::string& param) {
    if (param.empty()) return TelemetryOutput::None;
    if (param == "1") return TelemetryOutput::Stdout;
    if (param == "stdout") return TelemetryOutput::Stdout;
    if (param == "response") return TelemetryOutput::Response;

    // Parse comma-separated: "stdout,response"
    TelemetryOutput result = TelemetryOutput::None;
    std::stringstream ss(param);
    std::string token;
    while (std::getline(ss, token, ',')) {
        // Trim whitespace
        size_t start = token.find_first_not_of(" \t");
        size_t end = token.find_last_not_of(" \t");
        if (start != std::string::npos && end != std::string::npos) {
            token = token.substr(start, end - start + 1);
        }

        if (token == "stdout" || token == "1") {
            result = result | TelemetryOutput::Stdout;
        } else if (token == "response") {
            result = result | TelemetryOutput::Response;
        }
    }
    return result;
}
