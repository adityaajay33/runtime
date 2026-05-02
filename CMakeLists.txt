cmake_minimum_required(VERSION 3.16)

project(ptk LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(ptk_core
    src/pipeline/synthetic_frame_input.cc
    src/pipeline/console_target.cc
    src/pipeline/detection_decoder.cc
    src/pipeline/perception_pipeline.cc
    src/models/fake_model_engine.cc
)

target_include_directories(ptk_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(ptk_core
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_executable(ptk_fake_detection
    apps/ptk_fake_detection.cc
)

target_link_libraries(ptk_fake_detection
    PRIVATE
        ptk_core
)