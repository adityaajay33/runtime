#pragma once

#include "core/detection.h"
#include "core/frame.h"
#include "core/tensor.h"
#include "models/model_engine.h"
#include "pipeline/detection_decoder.h"
#include "pipeline/frame_input.h"
#include "pipeline/perception_target.h"

#include <cstddef>

namespace ptk
{

    struct PerceptionPipelineConfig
    {
        std::size_t max_frames = 300;
    };

    class PerceptionPipeline
    {
    public:
        PerceptionPipeline(
            FrameInput &input,
            ModelEngine &model,
            DetectionDecoder &decoder,
            PerceptionTarget &target,
            PerceptionPipelineConfig config = {});

        bool run();

    private:
        FrameInput &input_;
        ModelEngine &model_;
        DetectionDecoder &decoder_;
        PerceptionTarget &target_;
        PerceptionPipelineConfig config_;
    };

} // namespace ptk