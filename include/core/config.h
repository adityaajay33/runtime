#pragma once

#include "models/model_engine.h"
#include "pipeline/detection_decoder.h"
#include "pipeline/perception_pipeline.h"
#include "pipeline/synthetic_frame_input.h"

#include <string>

namespace ptk
{

    struct AppConfig
    {
        SyntheticFrameInputConfig input;
        ModelConfig model;
        DetectionDecoderConfig detection;
        PerceptionPipelineConfig pipeline;
    };

    // loads the MVP app config from a yaml file
    AppConfig load_config(const std::string &path);

} // namespace ptk