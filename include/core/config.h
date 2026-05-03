#pragma once

#include "models/model_engine.h"
#include "pipeline/display_target.h"
#include "pipeline/image_preprocessor.h"
#include "pipeline/video_frame_input.h"
#include "pipeline/yolo_detection_decoder.h"

#include <string>

namespace ptk
{

    struct AppConfig
    {
        std::string input_type = "video";
        std::string model_type = "onnx";
        std::string target_type = "display";

        VideoFrameInputConfig video_input;
        ModelConfig model;
        ImagePreprocessorConfig preprocess;
        YoloDetectionDecoderConfig detection;
        DisplayTargetConfig display_target;
    };

    AppConfig load_config(const std::string &path);

} // namespace ptk