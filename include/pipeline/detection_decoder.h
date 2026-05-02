#pragma once

#include "core/detection.h"
#include "core/frame.h"
#include "core/tensor.h"

#include <string>

namespace ptk
{

    struct DetectionDecoderConfig
    {
        float confidence_threshold = 0.5F;
        int target_class_id = 15;
        std::string target_class_name = "cat";
    };

    class DetectionDecoder
    {
    public:
        explicit DetectionDecoder(DetectionDecoderConfig config);

        bool run(const TensorView &model_output, const Frame &frame, PerceptionResult &result);

    private:
        DetectionDecoderConfig config_;
    };

} // namespace ptk