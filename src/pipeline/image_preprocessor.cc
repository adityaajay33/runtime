#include "pipeline/image_preprocessor.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>

namespace ptk
{

    ImagePreprocessor::ImagePreprocessor(ImagePreprocessorConfig config)
        : config_(config) {}

    bool ImagePreprocessor::run(const Frame &frame, TensorView output, PreprocessInfo &info)
    {
        if (!frame.valid() || !output.valid())
        {
            return false;
        }

        if (frame.pixel_format != PixelFormat::BGR8 && frame.pixel_format != PixelFormat::RGB8)
        {
            return false;
        }

        if (output.dtype != DataType::Float32 || output.layout != TensorLayout::NCHW)
        {
            return false;
        }

        if (output.shape.dims.size() != 4 ||
            output.shape.dims[0] != 1 ||
            output.shape.dims[1] != 3 ||
            output.shape.dims[2] != config_.model_height ||
            output.shape.dims[3] != config_.model_width)
        {
            return false;
        }

        auto *input_data = frame.image.data_as<uint8_t>();
        if (input_data == nullptr)
        {
            return false;
        }

        cv::Mat input(frame.height, frame.width, CV_8UC3, input_data);

        cv::Mat rgb;
        if (frame.pixel_format == PixelFormat::BGR8)
        {
            cv::cvtColor(input, rgb, cv::COLOR_BGR2RGB);
        }
        else
        {
            rgb = input;
        }

        const float scale_x = static_cast<float>(config_.model_width) /
                              static_cast<float>(frame.width);
        const float scale_y = static_cast<float>(config_.model_height) /
                              static_cast<float>(frame.height);
        const float scale = std::min(scale_x, scale_y);

        const int resized_width = static_cast<int>(std::round(frame.width * scale));
        const int resized_height = static_cast<int>(std::round(frame.height * scale));

        const int pad_x = (config_.model_width - resized_width) / 2;
        const int pad_y = (config_.model_height - resized_height) / 2;

        cv::Mat resized;
        cv::resize(rgb, resized, cv::Size(resized_width, resized_height));

        cv::Mat letterbox(
            config_.model_height,
            config_.model_width,
            CV_8UC3,
            cv::Scalar(114, 114, 114));

        resized.copyTo(
            letterbox(
                cv::Rect(pad_x, pad_y, resized_width, resized_height)));

        auto *output_data = output.data_as<float>();

        const int channels = 3;
        const int height = config_.model_height;
        const int width = config_.model_width;
        const float norm = config_.normalize ? (1.0F / 255.0F) : 1.0F;

        for (int c = 0; c < channels; ++c)
        {
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    const cv::Vec3b pixel = letterbox.at<cv::Vec3b>(y, x);
                    const int chw_index = c * height * width + y * width + x;
                    output_data[chw_index] = static_cast<float>(pixel[c]) * norm;
                }
            }
        }

        info.original_width = frame.width;
        info.original_height = frame.height;
        info.model_width = config_.model_width;
        info.model_height = config_.model_height;
        info.scale = scale;
        info.pad_x = pad_x;
        info.pad_y = pad_y;

        return true;
    }

} // namespace ptk