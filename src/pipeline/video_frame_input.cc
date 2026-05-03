#include "pipeline/video_frame_input.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <utility>

namespace ptk
{

    VideoFrameInput::VideoFrameInput(VideoFrameInputConfig config)
        : config_(std::move(config)) {}

    bool VideoFrameInput::open()
    {
        if (config_.path.empty())
        {
            return false;
        }

        capture_.open(config_.path);

        if (!capture_.isOpened())
        {
            return false;
        }

        sequence_ = 0;
        opened_ = true;

        return true;
    }

    bool VideoFrameInput::read_next(Frame &frame)
    {
        if (!opened_)
        {
            return false;
        }

        if (sequence_ >= config_.max_frames)
        {
            return false;
        }

        cv::Mat image;
        if (!capture_.read(image))
        {
            return false;
        }

        if (image.empty() || image.channels() != 3)
        {
            return false;
        }

        if (!copy_mat_to_tensor(image))
        {
            return false;
        }

        const auto now = std::chrono::steady_clock::now().time_since_epoch();
        const auto timestamp_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();

        frame.image = image_tensor_.view();
        frame.pixel_format = PixelFormat::BGR8;
        frame.width = image.cols;
        frame.height = image.rows;
        frame.channels = image.channels();
        frame.timestamp = Timestamp{timestamp_ns};
        frame.sequence = sequence_;
        frame.frame_id = config_.frame_id;
        frame.source_id = config_.source_id;

        ++sequence_;

        return true;
    }

    void VideoFrameInput::close()
    {
        if (capture_.isOpened())
        {
            capture_.release();
        }

        opened_ = false;
    }

    bool VideoFrameInput::copy_mat_to_tensor(const cv::Mat &image)
    {
        const int height = image.rows;
        const int width = image.cols;
        const int channels = image.channels();

        const TensorShape shape({static_cast<int64_t>(height),
                                 static_cast<int64_t>(width),
                                 static_cast<int64_t>(channels)});

        const std::size_t required_bytes = shape.size_bytes(DataType::UInt8);

        if (image_tensor_.size_bytes() != required_bytes)
        {
            image_tensor_ = Tensor(
                shape,
                DataType::UInt8,
                TensorLayout::HWC,
                Device::cpu());
        }

        auto view = image_tensor_.view();
        auto *dst = view.data_as<uint8_t>();

        if (image.isContinuous())
        {
            std::memcpy(dst, image.data, required_bytes);
            return true;
        }

        const std::size_t row_bytes = static_cast<std::size_t>(width * channels);

        for (int y = 0; y < height; ++y)
        {
            const auto *src_row = image.ptr<uint8_t>(y);
            auto *dst_row = dst + static_cast<std::size_t>(y) * row_bytes;
            std::memcpy(dst_row, src_row, row_bytes);
        }

        return true;
    }

} // namespace ptk