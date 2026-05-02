#pragma once

#include "core/frame.h"
#include "core/tensor.h"
#include "pipeline/frame_input.h"

#include <cstdint>
#include <string>

namespace ptk
{

    struct SyntheticFrameInputConfig
    {
        int width = 640;
        int height = 480;
        int channels = 3;
        uint64_t max_frames = 300;

        std::string frame_id = "camera";
        std::string source_id = "synthetic";
    };

    class SyntheticFrameInput : public FrameInput
    {
    public:
        explicit SyntheticFrameInput(SyntheticFrameInputConfig config);

        bool open() override;
        bool read_next(Frame &frame) override;
        void close() override;

    private:
        void fill_frame_pattern();

        SyntheticFrameInputConfig config_;
        uint64_t sequence_ = 0;
        bool opened_ = false;

        Tensor image_tensor_;
    };

} // namespace ptk