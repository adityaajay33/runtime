#include "pipeline/display_target.h"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <utility>

namespace ptk {

DisplayTarget::DisplayTarget(DisplayTargetConfig config)
    : config_(std::move(config)) {}

bool DisplayTarget::open() {
    cv::namedWindow(config_.window_name, cv::WINDOW_AUTOSIZE);
    opened_ = true;
    return true;
}

bool DisplayTarget::write(const PerceptionResult& result, const Frame& frame) {
    if (!opened_ || !frame.valid()) {
        return false;
    }

    cv::Mat image = make_frame_view(frame);

    if (image.empty()) {
        return false;
    }

    cv::Mat display_image = image;

    for (const auto& detection : result.detections) {
        draw_detection(display_image, detection);
    }

    cv::imshow(config_.window_name, display_image);

    const int key = cv::waitKey(config_.wait_key_ms);

    if (key == 27 || key == 'q') {
        return false;
    }

    return true;
}

void DisplayTarget::close() {
    if (opened_) {
        cv::destroyWindow(config_.window_name);
    }

    opened_ = false;
}

cv::Mat DisplayTarget::make_frame_view(const Frame& frame) {
    if (frame.pixel_format != PixelFormat::BGR8 &&
        frame.pixel_format != PixelFormat::RGB8 &&
        frame.pixel_format != PixelFormat::Gray8) {
        return {};
    }

    int cv_type = CV_8UC3;

    if (frame.pixel_format == PixelFormat::Gray8) {
        cv_type = CV_8UC1;
    }

    auto* data = frame.image.data_as<uint8_t>();

    if (data == nullptr) {
        return {};
    }

    // this cv::Mat does not own memory; it views the frame tensor memory
    cv::Mat image(frame.height, frame.width, cv_type, data);

    if (frame.pixel_format == PixelFormat::RGB8) {
        cv::Mat bgr;
        cv::cvtColor(image, bgr, cv::COLOR_RGB2BGR);
        return bgr;
    }

    return image;
}

void DisplayTarget::draw_detection(cv::Mat& image, const Detection2D& detection) {
    if (!detection.valid()) {
        return;
    }

    const int x = static_cast<int>(std::max(0.0F, detection.bbox.x));
    const int y = static_cast<int>(std::max(0.0F, detection.bbox.y));
    const int width = static_cast<int>(std::max(0.0F, detection.bbox.width));
    const int height = static_cast<int>(std::max(0.0F, detection.bbox.height));

    const cv::Rect box(x, y, width, height);

    cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);

    std::ostringstream label;
    label << detection.class_name << " "
          << std::fixed << std::setprecision(2)
          << detection.confidence;

    const cv::Point label_origin(x, std::max(20, y - 8));

    cv::putText(
        image,
        label.str(),
        label_origin,
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        cv::Scalar(0, 255, 0),
        2
    );
}

} // namespace ptk