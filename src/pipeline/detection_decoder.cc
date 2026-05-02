#include "pipeline/detection_decoder.h"

namespace ptk {

DetectionDecoder::DetectionDecoder(DetectionDecoderConfig config)
    : config_(std::move(config)) {}

bool DetectionDecoder::run(
    const TensorView& model_output,
    const Frame& frame,
    PerceptionResult& result
) {
    if (!model_output.valid() || !frame.valid()) {
        return false;
    }

    if (model_output.dtype != DataType::Float32 || model_output.num_elements() < 6) {
        return false;
    }

    const auto* data = model_output.data_as<float>();

    const float x = data[0];
    const float y = data[1];
    const float width = data[2];
    const float height = data[3];
    const float confidence = data[4];
    const int class_id = static_cast<int>(data[5]);

    result.detections.clear();
    result.timestamp = frame.timestamp;
    result.frame_sequence = frame.sequence;
    result.frame_id = frame.frame_id;
    result.source_id = frame.source_id;

    if (confidence < config_.confidence_threshold) {
        return true;
    }

    if (class_id != config_.target_class_id) {
        return true;
    }

    Detection2D detection;
    detection.class_id = class_id;
    detection.class_name = config_.target_class_name;
    detection.confidence = confidence;
    detection.bbox = BoundingBox2D{x, y, width, height};
    detection.timestamp = frame.timestamp;
    detection.frame_sequence = frame.sequence;
    detection.frame_id = frame.frame_id;
    detection.source_id = frame.source_id;

    if (detection.valid()) {
        result.detections.push_back(detection);
    }

    return true;
}

} // namespace ptk