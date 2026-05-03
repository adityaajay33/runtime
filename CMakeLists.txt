cmake_minimum_required(VERSION 3.16)

project(ptk LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)

# Prefer a bundled third_party install; fall back to any system install (e.g. Homebrew).
set(ONNXRUNTIME_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime"
    CACHE PATH "path to ONNX Runtime (leave empty to auto-detect)"
)

if(EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        PATHS "${ONNXRUNTIME_ROOT}/lib"
        NO_DEFAULT_PATH
        REQUIRED
    )
else()
    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        PATH_SUFFIXES onnxruntime
        REQUIRED
    )
    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        REQUIRED
    )
endif()

message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")

add_library(ptk_core
    src/core/config.cc

    src/pipeline/video_frame_input.cc
    src/pipeline/display_target.cc
    src/pipeline/image_preprocessor.cc
    src/pipeline/yolo_detection_decoder.cc

    src/models/onnx_model_engine.cc
)

target_include_directories(ptk_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${ONNXRUNTIME_INCLUDE_DIR}
)

target_compile_options(ptk_core
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_link_libraries(ptk_core
    PUBLIC
        yaml-cpp::yaml-cpp
        ${OpenCV_LIBS}
        ${ONNXRUNTIME_LIBRARY}
)

add_executable(ptk_yolo_detection
    apps/ptk_yolo_detection.cc
)

target_link_libraries(ptk_yolo_detection
    PRIVATE
        ptk_core
)