#pragma once

#include "core/detection.h"
#include "pipeline/perception_target.h"

#include <opencv2/opencv.hpp>

#include <string>

namespace ptk
{

    struct DisplayTargetConfig
    {
        std::string window_name = "ptk display";
        bool draw_fps = true;
        int wait_key_ms = 1;
    };

    class DisplayTarget : public PerceptionTarget
    {
    public:
        explicit DisplayTarget(DisplayTargetConfig config = {});

        bool open() override;
        bool write(const PerceptionResult &result, const Frame &frame) override;
        void close() override;

    private:
        cv::Mat make_frame_view(const Frame &frame);
        void draw_detection(cv::Mat &image, const Detection2D &detection);

        DisplayTargetConfig config_;
        bool opened_ = false;
    };

} // namespace ptk