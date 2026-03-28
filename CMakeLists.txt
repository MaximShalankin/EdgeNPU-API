cmake_minimum_required(VERSION 3.10)
project(EdgeNPU-API VERSION 1.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(EDGE_NPU_REQUIRE_OPENCV "Fail configure if OpenCV is missing" ON)
option(BUILD_TESTING "Build unit tests" ON)

# Paths (override with -D or environment)
set(RKNN_LLM_SDK "" CACHE PATH "Path to rknn-llm SDK root")

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/lib/rkllm
    ${CMAKE_SOURCE_DIR}/lib/rknn
    ${CMAKE_SOURCE_DIR}/lib/image_enc
)

# Bundled libraries
link_directories(${CMAKE_SOURCE_DIR}/lib/${CMAKE_SYSTEM_PROCESSOR})

# Optional: external SDK paths (overrides bundled lib/)
if(RKNN_LLM_SDK)
    include_directories(
        ${RKNN_LLM_SDK}/rkllm-runtime/Linux/librkllm_api/include
        ${RKNN_LLM_SDK}/examples/multimodal_model_demo/deploy/src
        ${RKNN_LLM_SDK}/examples/multimodal_model_demo/deploy/3rdparty/librknnrt/Linux/librknn_api/include
    )
    link_directories(
        ${RKNN_LLM_SDK}/rkllm-runtime/Linux/librkllm_api/lib
        ${RKNN_LLM_SDK}/examples/multimodal_model_demo/deploy/3rdparty/librknnrt/Linux/librknn_api/lib
    )
endif()

# OpenCV - try pkg-config first, then find_package
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(OpenCV QUIET opencv4)
endif()
if(NOT OpenCV_FOUND)
    find_package(OpenCV QUIET)
endif()
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    message(STATUS "OpenCV found: ${OpenCV_INCLUDE_DIRS}")
else()
    if(EDGE_NPU_REQUIRE_OPENCV)
        message(FATAL_ERROR "OpenCV not found. Install OpenCV (opencv4 headers/libs) or configure with -DEDGE_NPU_REQUIRE_OPENCV=OFF if you know what you're doing.")
    endif()
    # Best-effort fallback to common system path (may still fail at build time)
    include_directories(/usr/include/opencv4)
    set(OpenCV_LIBS opencv_core opencv_imgproc opencv_imgcodecs)
endif()

# Source files
set(SOURCES
    src/main_server.cpp
    lib/image_enc/image_enc.cc
)

# Executable
add_executable(rkllm-server ${SOURCES})

# Link libraries
if(OpenCV_FOUND AND OpenCV_LIBRARIES)
    set(OPENCV_LINK_LIBS ${OpenCV_LIBRARIES})
else()
    set(OPENCV_LINK_LIBS opencv_core opencv_imgproc opencv_imgcodecs)
endif()

target_link_libraries(rkllm-server
    rkllmrt
    rknnrt
    ${OPENCV_LINK_LIBS}
    pthread
)

# Install
install(TARGETS rkllm-server DESTINATION bin)

# Unit tests
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
