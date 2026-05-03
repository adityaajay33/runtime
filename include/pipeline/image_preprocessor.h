#pragma once

#include "core/frame.h"
#include "core/tensor.h"

namespace ptk
{

    struct PreprocessInfo
    {
        int original_width = 0;
        int original_height = 0;

        int model_width = 0;
        int model_height = 0;

        float scale = 1.0F;
        int pad_x = 0;
        int pad_y = 0;
    };

    struct ImagePreprocessorConfig
    {
        int model_width = 640;
        int model_height = 640;
        bool normalize = true;
    };

    class ImagePreprocessor
    {
    public:
        explicit ImagePreprocessor(ImagePreprocessorConfig config);

        bool run(const Frame &frame, TensorView output, PreprocessInfo &info);

    private:
        ImagePreprocessorConfig config_;
    };

} // namespace ptk