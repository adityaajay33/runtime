#pragma once

#include "core/frame.h"
#include "core/tensor.h"
#include "pipeline/frame_input.h"

#include <opencv2/opencv.hpp>

#include <cstdint>
#include <string>

namespace ptk
{

    struct VideoFrameInputConfig
    {
        std::string path;
        std::string frame_id = "camera";
        std::string source_id = "video";

        uint64_t max_frames = 300;
    };

    class VideoFrameInput : public FrameInput
    {
    public:
        explicit VideoFrameInput(VideoFrameInputConfig config);

        bool open() override;
        bool read_next(Frame &frame) override;
        void close() override;

    private:
        bool copy_mat_to_tensor(const cv::Mat &image);

        VideoFrameInputConfig config_;
        cv::VideoCapture capture_;

        uint64_t sequence_ = 0;
        bool opened_ = false;

        Tensor image_tensor_;
    };

} // namespace ptk