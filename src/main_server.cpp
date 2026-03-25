// VLM HTTP Server - Qwen3-VL Persistent API Server
// Based on rknn-llm multimodal demo with cpp-httplib integration
//
// Copyright (c) 2025 by Rockchip Electronics Co., Ltd. All Rights Reserved.
// HTTP Server extension for persistent model serving

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <mutex>
#include <sstream>
#include <vector>
#include <signal.h>
#include <opencv2/opencv.hpp>

#include "image_enc.h"
#include "rkllm.h"
#include "httplib.h"
#include "json.hpp"
#include "base64.h"

using namespace std;
using json = nlohmann::json;

// Model type enumeration
enum class ModelType {
    VLM,    // Vision + LLM (Qwen3-VL, MiniCPM-V)
    LLM     // Text only (Qwen2.5, Qwen3, Llama-3.2)
};

// Convert string to ModelType
ModelType string_to_model_type(const string& s) {
    if (s == "llm" || s == "LLM") return ModelType::LLM;
    return ModelType::VLM;  // Default to VLM
}

// Convert ModelType to string
string model_type_to_string(ModelType t) {
    return (t == ModelType::LLM) ? "llm" : "vlm";
}

// Prompt family enumeration (for different chat templates)
enum class PromptFamily {
    QWEN,   // Qwen2.5, Qwen3, Qwen3-VL: <|im_start|>user\n...<|im_end|>
    LLAMA   // Llama-3.2: <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n...<|eot_id|>
};

// Convert string to PromptFamily
PromptFamily string_to_prompt_family(const string& s) {
    if (s == "llama" || s == "LLAMA") return PromptFamily::LLAMA;
    return PromptFamily::QWEN;  // Default to Qwen
}

// Convert PromptFamily to string
string prompt_family_to_string(PromptFamily f) {
    return (f == PromptFamily::LLAMA) ? "llama" : "qwen";
}

// Format prompt for Qwen models
string format_qwen_prompt(const string& user_content) {
    return "<|im_start|>user\n" + user_content + "<|im_end|>\n<|im_start|>assistant\n";
}

// Format prompt for Llama models
string format_llama_prompt(const string& user_content) {
    return "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" +
           user_content + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
}

// Global state
LLMHandle llmHandle = nullptr;
rknn_app_context_t rknn_app_ctx;
bool models_loaded = false;
bool vision_loaded = false;
std::mutex npu_mutex;

// Model configuration
struct ModelConfig {
    string llm_path;
    string vision_path;
    int max_new_tokens;
    int max_context_len;
    int npu_core_num;
    int port;
    ModelType type;
    PromptFamily prompt_family;  // Prompt template family (Qwen or Llama)
    string model_name;
} config;

// Response accumulator for streaming
struct ResponseAccumulator {
    string text;
    bool finished;
    bool error;
};

ResponseAccumulator global_response;

// Signal handler for graceful shutdown
void signal_handler(int signal) {
    cout << "\n[Server] Shutting down..." << endl;

    if (models_loaded) {
        // Release vision encoder only if loaded (VLM mode)
        if (vision_loaded) {
            release_imgenc(&rknn_app_ctx);
            vision_loaded = false;
        }
        if (llmHandle != nullptr) {
            LLMHandle _tmp = llmHandle;
            llmHandle = nullptr;
            rkllm_destroy(_tmp);
        }
        models_loaded = false;
    }

    exit(signal);
}

// RKLLM callback - accumulates response text
int callback(RKLLMResult *result, void *userdata, LLMCallState state) {
    if (state == RKLLM_RUN_FINISH) {
        global_response.finished = true;
    } else if (state == RKLLM_RUN_ERROR) {
        global_response.error = true;
        global_response.finished = true;
    } else if (state == RKLLM_RUN_NORMAL) {
        global_response.text += string(result->text);
    }
    return 0;
}

// Expand image to square with background color
cv::Mat expand2square(const cv::Mat& img, const cv::Scalar& background_color) {
    int width = img.cols;
    int height = img.rows;

    if (width == height) {
        return img.clone();
    }

    int size = std::max(width, height);
    cv::Mat result(size, size, img.type(), background_color);

    int x_offset = (size - width) / 2;
    int y_offset = (size - height) / 2;

    cv::Rect roi(x_offset, y_offset, width, height);
    img.copyTo(result(roi));

    return result;
}

// Decode base64 image to cv::Mat
bool decode_base64_image(const string& base64_data, cv::Mat& out_image) {
    try {
        // Find data prefix if present
        string base64_str = base64_data;
        size_t comma_pos = base64_data.find(",");
        if (comma_pos != string::npos) {
            base64_str = base64_data.substr(comma_pos + 1);
        }

        // Decode base64
        string decoded = base64::from_base64(base64_str);

        // Decode image via OpenCV
        vector<uint8_t> img_data(decoded.begin(), decoded.end());
        out_image = cv::imdecode(img_data, cv::IMREAD_COLOR);

        return !out_image.empty();
    } catch (const exception& e) {
        cerr << "[Error] Base64 decode failed: " << e.what() << endl;
        return false;
    }
}

// Run text-only LLM inference (no vision)
string run_llm_inference(const string& prompt) {
    // Reset response accumulator
    global_response.text = "";
    global_response.finished = false;
    global_response.error = false;

    // Format prompt according to model family
    string formatted_prompt;
    if (config.prompt_family == PromptFamily::LLAMA) {
        formatted_prompt = format_llama_prompt(prompt);
    } else {
        formatted_prompt = format_qwen_prompt(prompt);
    }

    // Prepare RKLLM input
    RKLLMInput rkllm_input;
    memset(&rkllm_input, 0, sizeof(RKLLMInput));

    RKLLMInferParam rkllm_infer_params;
    memset(&rkllm_infer_params, 0, sizeof(RKLLMInferParam));

    rkllm_infer_params.mode = RKLLM_INFER_GENERATE;
    rkllm_infer_params.keep_history = 0;

    rkllm_input.input_type = RKLLM_INPUT_PROMPT;
    rkllm_input.role = "user";
    rkllm_input.prompt_input = (char*)formatted_prompt.c_str();

    // Run inference
    rkllm_run(llmHandle, &rkllm_input, &rkllm_infer_params, NULL);

    if (global_response.error) {
        return "Error: Inference failed";
    }

    return global_response.text;
}

// Process image and run VLM inference
string run_vlm_inference(const cv::Mat& input_img, const string& prompt) {
    // Reset response accumulator
    global_response.text = "";
    global_response.finished = false;
    global_response.error = false;

    // Convert BGR to RGB
    cv::Mat img;
    cv::cvtColor(input_img, img, cv::COLOR_BGR2RGB);

    // Expand to square
    cv::Scalar background_color(127.5, 127.5, 127.5);
    cv::Mat square_img = expand2square(img, background_color);

    // Resize to model input size
    size_t image_width = rknn_app_ctx.model_width;
    size_t image_height = rknn_app_ctx.model_height;
    cv::Mat resized_img;
    cv::resize(square_img, resized_img, cv::Size(image_width, image_height), 0, 0, cv::INTER_LINEAR);

    // Run image encoder
    size_t n_image_tokens = rknn_app_ctx.model_image_token;
    size_t image_embed_len = rknn_app_ctx.model_embed_size;
    size_t n_embed_output = rknn_app_ctx.io_num.n_output;
    int rkllm_image_embed_len = n_image_tokens * image_embed_len * n_embed_output;

    float* img_vec = new float[rkllm_image_embed_len];
    memset(img_vec, 0, rkllm_image_embed_len * sizeof(float));

    int ret = run_imgenc(&rknn_app_ctx, resized_img.data, img_vec);
    if (ret != 0) {
        delete[] img_vec;
        return "Error: Image encoding failed";
    }

    // Prepare RKLLM input
    RKLLMInput rkllm_input;
    memset(&rkllm_input, 0, sizeof(RKLLMInput));

    RKLLMInferParam rkllm_infer_params;
    memset(&rkllm_infer_params, 0, sizeof(RKLLMInferParam));

    rkllm_infer_params.mode = RKLLM_INFER_GENERATE;
    rkllm_infer_params.keep_history = 0;

    // Check if prompt contains <image> tag
    string full_prompt = prompt;
    if (prompt.find("<image>") == string::npos) {
        full_prompt = "<image>" + prompt;
    }

    rkllm_input.input_type = RKLLM_INPUT_MULTIMODAL;
    rkllm_input.role = "user";
    rkllm_input.multimodal_input.prompt = (char*)full_prompt.c_str();
    rkllm_input.multimodal_input.image_embed = img_vec;
    rkllm_input.multimodal_input.n_image_tokens = n_image_tokens;
    rkllm_input.multimodal_input.n_image = 1;
    rkllm_input.multimodal_input.image_height = image_height;
    rkllm_input.multimodal_input.image_width = image_width;

    // Run inference
    rkllm_run(llmHandle, &rkllm_input, &rkllm_infer_params, NULL);

    delete[] img_vec;

    if (global_response.error) {
        return "Error: Inference failed";
    }

    return global_response.text;
}

// Initialize models
bool init_models() {
    auto t_start = chrono::high_resolution_clock::now();

    // Initialize LLM
    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = config.llm_path.c_str();
    param.top_k = 1;
    param.max_new_tokens = config.max_new_tokens;
    param.max_context_len = config.max_context_len;
    param.skip_special_token = true;
    param.extend_param.base_domain_id = 1;

    // Set vision tokens only for VLM mode
    if (config.type == ModelType::VLM) {
        param.img_start = "<|vision_start|>";
        param.img_end = "<|vision_end|>";
        param.img_content = "<|image_pad|>";
    }

    int ret = rkllm_init(&llmHandle, &param, callback);
    if (ret != 0) {
        cerr << "[Error] rkllm_init failed" << endl;
        return false;
    }

    auto t_llm = chrono::high_resolution_clock::now();
    auto llm_time = chrono::duration_cast<chrono::milliseconds>(t_llm - t_start);
    cout << "[Init] LLM loaded in " << llm_time.count() << " ms" << endl;

    // Initialize vision encoder only for VLM mode
    if (config.type == ModelType::VLM) {
        memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
        ret = init_imgenc(config.vision_path.c_str(), &rknn_app_ctx, config.npu_core_num);
        if (ret != 0) {
            cerr << "[Error] init_imgenc failed" << endl;
            rkllm_destroy(llmHandle);
            llmHandle = nullptr;
            return false;
        }

        auto t_vision = chrono::high_resolution_clock::now();
        auto vision_time = chrono::duration_cast<chrono::milliseconds>(t_vision - t_llm);
        cout << "[Init] Vision encoder loaded in " << vision_time.count() << " ms" << endl;
        vision_loaded = true;
    } else {
        cout << "[Init] Skipping vision encoder (LLM mode)" << endl;
        vision_loaded = false;
    }

    models_loaded = true;
    return true;
}

// Generate OpenAI-compatible response ID
string generate_response_id() {
    static int counter = 0;
    char buf[64];
    snprintf(buf, sizeof(buf), "chatcmpl-%d%08x", ++counter, (unsigned int)time(NULL));
    return string(buf);
}

// Setup HTTP routes
void setup_routes(httplib::Server& svr) {
    // Root endpoint
    svr.Get("/", [](const httplib::Request& req, httplib::Response& res) {
        json response = {
            {"name", "LLM HTTP Server"},
            {"version", "1.1.0"},
            {"model", config.model_name},
            {"type", model_type_to_string(config.type)},
            {"mode", "persistent"}
        };
        res.set_content(response.dump(), "application/json");
    });

    // Health check
    svr.Get("/health", [](const httplib::Request& req, httplib::Response& res) {
        json response = {
            {"status", models_loaded ? "healthy" : "unhealthy"},
            {"model", config.model_name},
            {"type", model_type_to_string(config.type)},
            {"llm_loaded", models_loaded},
            {"vision_loaded", vision_loaded},
            {"mode", "persistent"}
        };
        res.set_content(response.dump(), "application/json");
    });

    // List models (OpenAI-compatible)
    svr.Get("/v1/models", [](const httplib::Request& req, httplib::Response& res) {
        json response = {
            {"object", "list"},
            {"data", json::array({
                {
                    {"id", config.model_name},
                    {"object", "model"},
                    {"created", 1700000000},
                    {"owned_by", "local"}
                }
            })}
        };
        res.set_content(response.dump(), "application/json");
    });

    // Get model info (OpenAI-compatible)
    svr.Get(R"(/v1/models/(.+))", [](const httplib::Request& req, httplib::Response& res) {
        string model_id = req.matches[1];

        if (model_id != config.model_name) {
            json error = {
                {"error", {
                    {"message", "Model " + model_id + " not found"},
                    {"type", "invalid_request_error"}
                }}
            };
            res.status = 404;
            res.set_content(error.dump(), "application/json");
            return;
        }

        json response = {
            {"id", config.model_name},
            {"object", "model"},
            {"created", 1700000000},
            {"owned_by", "local"}
        };
        res.set_content(response.dump(), "application/json");
    });

    // Chat completions (OpenAI-compatible)
    svr.Post("/v1/chat/completions", [](const httplib::Request& req, httplib::Response& res) {
        if (!models_loaded) {
            json error = {
                {"error", {
                    {"message", "Models not loaded"},
                    {"type", "server_error"}
                }}
            };
            res.status = 503;
            res.set_content(error.dump(), "application/json");
            return;
        }

        try {
            json body = json::parse(req.body);

            // Parse messages
            if (!body.contains("messages") || !body["messages"].is_array()) {
                json error = {
                    {"error", {
                        {"message", "No messages provided"},
                        {"type", "invalid_request_error"}
                    }}
                };
                res.status = 400;
                res.set_content(error.dump(), "application/json");
                return;
            }

            string user_content;
            string image_data;

            for (const auto& msg : body["messages"]) {
                if (msg["role"] == "user") {
                    if (msg["content"].is_string()) {
                        user_content = msg["content"];
                    } else if (msg["content"].is_array()) {
                        // Multimodal content
                        for (const auto& item : msg["content"]) {
                            if (item["type"] == "text") {
                                user_content += item["text"].get<string>();
                            } else if (item["type"] == "image_url") {
                                if (item.contains("image_url") && item["image_url"].contains("url")) {
                                    image_data = item["image_url"]["url"];
                                }
                            }
                        }
                    }
                }
            }

            // Run inference with thread safety
            string result;

            if (config.type == ModelType::LLM) {
                // Text-only LLM inference
                lock_guard<mutex> lock(npu_mutex);
                cout << "[Server] Running LLM inference (text-only)..." << endl;
                auto t_start = chrono::high_resolution_clock::now();

                result = run_llm_inference(user_content);

                auto t_end = chrono::high_resolution_clock::now();
                auto elapsed = chrono::duration_cast<chrono::milliseconds>(t_end - t_start);
                cout << "[Server] LLM inference completed in " << elapsed.count() << " ms" << endl;
            } else {
                // VLM inference - requires image
                if (image_data.empty()) {
                    json error = {
                        {"error", {
                            {"message", "VLM model requires an image. Please provide an image in image_url format."},
                            {"type", "invalid_request_error"}
                        }}
                    };
                    res.status = 400;
                    res.set_content(error.dump(), "application/json");
                    return;
                }

                // Decode image
                cv::Mat img;
                if (!decode_base64_image(image_data, img)) {
                    json error = {
                        {"error", {
                            {"message", "Failed to decode image"},
                            {"type", "invalid_request_error"}
                        }}
                    };
                    res.status = 400;
                    res.set_content(error.dump(), "application/json");
                    return;
                }

                lock_guard<mutex> lock(npu_mutex);
                cout << "[Server] Running VLM inference..." << endl;
                auto t_start = chrono::high_resolution_clock::now();

                result = run_vlm_inference(img, user_content);

                auto t_end = chrono::high_resolution_clock::now();
                auto elapsed = chrono::duration_cast<chrono::milliseconds>(t_end - t_start);
                cout << "[Server] VLM inference completed in " << elapsed.count() << " ms" << endl;
            }

            // Format OpenAI response
            json response = {
                {"id", generate_response_id()},
                {"object", "chat.completion"},
                {"created", time(NULL)},
                {"model", config.model_name},
                {"choices", json::array({
                    {
                        {"index", 0},
                        {"message", {
                            {"role", "assistant"},
                            {"content", result}
                        }},
                        {"finish_reason", "stop"}
                    }
                })},
                {"usage", {
                    {"prompt_tokens", 0},
                    {"completion_tokens", 0},
                    {"total_tokens", 0}
                }}
            };

            res.set_content(response.dump(), "application/json");

        } catch (const json::exception& e) {
            json error = {
                {"error", {
                    {"message", string("JSON parse error: ") + e.what()},
                    {"type", "invalid_request_error"}
                }}
            };
            res.status = 400;
            res.set_content(error.dump(), "application/json");
        } catch (const exception& e) {
            json error = {
                {"error", {
                    {"message", string("Server error: ") + e.what()},
                    {"type", "server_error"}
                }}
            };
            res.status = 500;
            res.set_content(error.dump(), "application/json");
        }
    });
}

void print_usage(const char* prog) {
    cerr << "Usage: " << prog << " [options]\n"
         << "Options:\n"
         << "  --llm <path>          Path to LLM model (default: ~/models/qwen3-vl-4b-instruct_w8a8_rk3588.rkllm)\n"
         << "  --vision <path>       Path to vision encoder (default: ~/models/qwen3-vl_vision_rk3588.rknn)\n"
         << "  --type <type>         Model type: 'vlm' or 'llm' (default: vlm)\n"
         << "  --prompt-family <f>   Prompt format: 'qwen' or 'llama' (default: qwen)\n"
         << "  --name <name>         Model name for API responses (default: qwen3-vl)\n"
         << "  --port <num>          HTTP server port (default: 8082)\n"
         << "  --max-tokens <num>    Max new tokens (default: 256)\n"
         << "  --context-len <num>   Max context length (default: 512)\n"
         << "  --npu-cores <num>     NPU core count (default: 3)\n"
         << "  --help                Show this help\n"
         << "\nExamples:\n"
         << "  # VLM mode (with vision):\n"
         << "  " << prog << " --type vlm --llm ~/models/qwen3-vl.rkllm --vision ~/models/qwen3-vl_vision.rknn\n"
         << "\n"
         << "  # LLM mode (text-only, Qwen):\n"
         << "  " << prog << " --type llm --name qwen3-4b --llm ~/models/qwen3-4b.rkllm\n"
         << "\n"
         << "  # LLM mode (text-only, Llama):\n"
         << "  " << prog << " --type llm --name llama-3.2-1b --prompt-family llama --llm ~/models/llama32-1b.rkllm\n";
}

int main(int argc, char** argv) {
    // Default configuration
    config.llm_path = string(getenv("HOME")) + "/models/qwen3-vl-4b-instruct_w8a8_rk3588.rkllm";
    config.vision_path = string(getenv("HOME")) + "/models/qwen3-vl_vision_rk3588.rknn";
    config.max_new_tokens = 256;
    config.max_context_len = 512;
    config.npu_core_num = 3;
    config.port = 8082;
    config.type = ModelType::VLM;  // Default to VLM
    config.prompt_family = PromptFamily::QWEN;  // Default to Qwen prompt format
    config.model_name = "qwen3-vl";

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--llm" && i + 1 < argc) {
            config.llm_path = argv[++i];
        } else if (arg == "--vision" && i + 1 < argc) {
            config.vision_path = argv[++i];
        } else if (arg == "--type" && i + 1 < argc) {
            config.type = string_to_model_type(argv[++i]);
        } else if (arg == "--prompt-family" && i + 1 < argc) {
            config.prompt_family = string_to_prompt_family(argv[++i]);
        } else if (arg == "--name" && i + 1 < argc) {
            config.model_name = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            config.port = atoi(argv[++i]);
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            config.max_new_tokens = atoi(argv[++i]);
        } else if (arg == "--context-len" && i + 1 < argc) {
            config.max_context_len = atoi(argv[++i]);
        } else if (arg == "--npu-cores" && i + 1 < argc) {
            config.npu_core_num = atoi(argv[++i]);
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            cerr << "Unknown argument: " << arg << endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // Setup signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    cout << "========================================" << endl;
    cout << "LLM HTTP Server (Persistent Mode)" << endl;
    cout << "========================================" << endl;
    cout << "\nConfiguration:" << endl;
    cout << "  Model type: " << model_type_to_string(config.type) << endl;
    cout << "  Prompt family: " << prompt_family_to_string(config.prompt_family) << endl;
    cout << "  Model name: " << config.model_name << endl;
    cout << "  LLM: " << config.llm_path << endl;
    if (config.type == ModelType::VLM) {
        cout << "  Vision: " << config.vision_path << endl;
    }
    cout << "  Port: " << config.port << endl;
    cout << "  Max tokens: " << config.max_new_tokens << endl;
    cout << "  NPU cores: " << config.npu_core_num << endl;
    cout << endl;

    // Initialize models
    cout << "[Init] Loading models..." << endl;
    auto t_start = chrono::high_resolution_clock::now();

    if (!init_models()) {
        cerr << "[Error] Failed to initialize models" << endl;
        return 1;
    }

    auto t_end = chrono::high_resolution_clock::now();
    auto total_time = chrono::duration_cast<chrono::milliseconds>(t_end - t_start);
    cout << "[Init] All models loaded in " << total_time.count() << " ms" << endl;

    // Setup HTTP server
    httplib::Server svr;
    setup_routes(svr);

    cout << "\n========================================" << endl;
    cout << "Server ready!" << endl;
    cout << "========================================" << endl;
    cout << "\nEndpoints:" << endl;
    cout << "  GET  /              - Server info" << endl;
    cout << "  GET  /health        - Health check" << endl;
    cout << "  GET  /v1/models     - List models" << endl;
    if (config.type == ModelType::VLM) {
        cout << "  POST /v1/chat/completions - Chat with image" << endl;
    } else {
        cout << "  POST /v1/chat/completions - Chat (text-only)" << endl;
    }
    cout << "\nStarting HTTP server on port " << config.port << "..." << endl;

    // Start server
    if (!svr.listen("0.0.0.0", config.port)) {
        cerr << "[Error] Failed to start server on port " << config.port << endl;
        return 1;
    }

    return 0;
}
