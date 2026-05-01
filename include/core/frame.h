#pragma once

#include "core/tensor.h"

#include <cstdint>
#include <string>

namespace ptk
{
    enum class PixelFormat
    {
        Unknown,
        RGB8,
        BGR8,
        Gray8
    };

    struct Timestamp
    {
        int64_t ns = 0;
    };

    struct Frame
    {
        TensorView image;

        PixelFormat pixel_format = PixelFormat::Unknown;

        int width = 0;
        int height = 0;
        int channels = 0;

        Timestamp timestamp;
        uint64_t sequence = 0;

        std::string frame_id = "camera";
        std::string source_id = "unknown";

        bool valid() const
        {
            return image.valid() &&
                   width > 0 &&
                   height > 0 &&
                   channels > 0;
        }
    };

} // namespace ptk