# Unit tests for EdgeNPU-API
# Uses Catch2 via FetchContent (no system installation required)

include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.7.1
)
FetchContent_MakeAvailable(Catch2)

# Test executable
add_executable(unit_tests
    unit/test_main.cpp
)

target_include_directories(unit_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

target_link_libraries(unit_tests PRIVATE
    Catch2::Catch2WithMain
)

# Register with CTest
include(CTest)
include(Catch)
catch_discover_tests(unit_tests)
