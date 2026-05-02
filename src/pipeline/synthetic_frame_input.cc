#include "pipeline/synthetic_frame_input.h"

#include <chrono>
#include <cstdint>

namespace ptk
{

    SyntheticFrameInput::SyntheticFrameInput(SyntheticFrameInputConfig config)
        : config_(std::move(config)) {}

    bool SyntheticFrameInput::open()
    {
        if (config_.width <= 0 || config_.height <= 0 || config_.channels <= 0)
        {
            return false;
        }

        image_tensor_ = Tensor(
            TensorShape({static_cast<int64_t>(config_.height),
                         static_cast<int64_t>(config_.width),
                         static_cast<int64_t>(config_.channels)}),
            DataType::UInt8,
            TensorLayout::HWC,
            Device::cpu());

        sequence_ = 0;
        opened_ = true;

        return true;
    }

    bool SyntheticFrameInput::read_next(Frame &frame)
    {
        if (!opened_)
        {
            return false;
        }

        if (sequence_ >= config_.max_frames)
        {
            return false;
        }

        fill_frame_pattern();

        const auto now = std::chrono::steady_clock::now().time_since_epoch();
        const auto timestamp_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();

        frame.image = image_tensor_.view();
        frame.pixel_format = PixelFormat::RGB8;
        frame.width = config_.width;
        frame.height = config_.height;
        frame.channels = config_.channels;
        frame.timestamp = Timestamp{timestamp_ns};
        frame.sequence = sequence_;
        frame.frame_id = config_.frame_id;
        frame.source_id = config_.source_id;

        ++sequence_;

        return true;
    }

    void SyntheticFrameInput::close()
    {
        opened_ = false;
    }

    void SyntheticFrameInput::fill_frame_pattern()
    {
        auto view = image_tensor_.view();
        auto *data = view.data_as<uint8_t>();

        const int width = config_.width;
        const int height = config_.height;
        const int channels = config_.channels;

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const int base = (y * width + x) * channels;

                const uint8_t r = static_cast<uint8_t>((x + sequence_) % 256);
                const uint8_t g = static_cast<uint8_t>((y + sequence_) % 256);
                const uint8_t b = static_cast<uint8_t>((x + y + sequence_) % 256);

                if (channels >= 1)
                {
                    data[base + 0] = r;
                }

                if (channels >= 2)
                {
                    data[base + 1] = g;
                }

                if (channels >= 3)
                {
                    data[base + 2] = b;
                }
            }
        }
    }

} // namespace ptk