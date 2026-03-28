// Unit tests for ticket-005 telemetry system
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "telemetry.h"

using Catch::Matchers::WithinRel;

TEST_CASE("Parse telemetry output param", "[telemetry]") {
    REQUIRE(parse_telemetry_param("") == TelemetryOutput::None);
    REQUIRE(parse_telemetry_param("1") == TelemetryOutput::Stdout);
    REQUIRE(parse_telemetry_param("stdout") == TelemetryOutput::Stdout);
    REQUIRE(parse_telemetry_param("response") == TelemetryOutput::Response);
    REQUIRE(parse_telemetry_param("stdout,response") ==
            (TelemetryOutput::Stdout | TelemetryOutput::Response));
}

TEST_CASE("Parse telemetry with whitespace", "[telemetry]") {
    REQUIRE(parse_telemetry_param("stdout, response") ==
            (TelemetryOutput::Stdout | TelemetryOutput::Response));
    REQUIRE(parse_telemetry_param(" stdout ") == TelemetryOutput::Stdout);
}

TEST_CASE("Has output flag", "[telemetry]") {
    auto combined = TelemetryOutput::Stdout | TelemetryOutput::Response;
    REQUIRE(has_telemetry_output(combined, TelemetryOutput::Stdout));
    REQUIRE(has_telemetry_output(combined, TelemetryOutput::Response));
    REQUIRE_FALSE(has_telemetry_output(TelemetryOutput::Stdout, TelemetryOutput::Response));
    REQUIRE_FALSE(has_telemetry_output(TelemetryOutput::None, TelemetryOutput::Stdout));
}

TEST_CASE("InferenceMetrics decode_tokens_per_sec", "[telemetry]") {
    InferenceMetrics m;

    SECTION("Zero decode_ms returns 0") {
        m.output_tokens = 50;
        m.decode_ms = 0;
        REQUIRE(m.decode_tokens_per_sec() == 0.0f);
    }

    SECTION("Normal calculation") {
        m.output_tokens = 50;
        m.decode_ms = 200;
        REQUIRE_THAT(m.decode_tokens_per_sec(), WithinRel(250.0f, 0.01f));
    }
}

TEST_CASE("InferenceMetrics to_json", "[telemetry]") {
    InferenceMetrics m;
    m.preprocess_ms = 10;
    m.vision_encoder_ms = 250;
    m.prefill_ms = 45;
    m.decode_ms = 200;
    m.total_ms = 500;
    m.input_tokens = 49;
    m.output_tokens = 50;

    json j = m.to_json();

    REQUIRE(j["preprocess_ms"] == 10);
    REQUIRE(j["vision_encoder_ms"] == 250);
    REQUIRE(j["prefill_ms"] == 45);
    REQUIRE(j["decode_ms"] == 200);
    REQUIRE(j["total_ms"] == 500);
    REQUIRE(j["input_tokens"] == 49);
    REQUIRE(j["output_tokens"] == 50);
    REQUIRE_THAT(j["decode_tokens_per_sec"].get<float>(), WithinRel(250.0f, 0.01f));
}

TEST_CASE("TelemetryOutput bitwise OR", "[telemetry]") {
    auto result = TelemetryOutput::Stdout | TelemetryOutput::Response;

    REQUIRE(static_cast<uint8_t>(result) == 3);  // 0b00000011
    REQUIRE(has_telemetry_output(result, TelemetryOutput::Stdout));
    REQUIRE(has_telemetry_output(result, TelemetryOutput::Response));
}
