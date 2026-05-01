#pragma once

#include "core/frame.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ptk
{

    struct BoundingBox2D
    {
        float x = 0.0F;
        float y = 0.0F;
        float width = 0.0F;
        float height = 0.0F;

        bool valid() const
        {
            return width > 0.0F && height > 0.0F;
        }
    };

    struct Detection2D
    {
        int class_id = -1;
        std::string class_name;
        float confidence = 0.0F;

        BoundingBox2D bbox;

        Timestamp timestamp;
        uint64_t frame_sequence = 0;

        std::string frame_id = "camera";
        std::string source_id = "unknown";

        bool valid() const
        {
            return class_id >= 0 &&
                   confidence >= 0.0F &&
                   confidence <= 1.0F &&
                   bbox.valid();
        }
    };

    // all perception outputs produced from one input frame
    struct PerceptionResult
    {
        std::vector<Detection2D> detections;

        Timestamp timestamp;
        uint64_t frame_sequence = 0;

        std::string frame_id = "camera";
        std::string source_id = "unknown";

        bool empty() const
        {
            return detections.empty();
        }
    };

} // namespace ptk