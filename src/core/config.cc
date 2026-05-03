#include "core/config.h"

#include <yaml-cpp/yaml.h>

#include <stdexcept>
#include <string>

namespace ptk
{

    namespace
    {

        template <typename T>
        T get_or_default(const YAML::Node &node, const std::string &key, const T &default_value)
        {
            if (!node || !node[key])
            {
                return default_value;
            }

            return node[key].as<T>();
        }

        Device parse_device(const YAML::Node &node)
        {
            const std::string device = get_or_default<std::string>(node, "device", "cpu");

            if (device == "cpu")
            {
                return Device::cpu();
            }

            if (device == "cuda")
            {
                return Device::cuda(0);
            }

            if (device == "mps")
            {
                return Device::mps();
            }

            throw std::runtime_error("unsupported device in config: " + device);
        }

    } // namespace

    AppConfig load_config(const std::string &path)
    {
        YAML::Node root = YAML::LoadFile(path);

        AppConfig config;

        const YAML::Node input = root["input"];
        config.input_type = get_or_default<std::string>(input, "type", "video");
        config.video_input.path = get_or_default<std::string>(input, "path", "data/kitten.mp4");
        config.video_input.max_frames = get_or_default<uint64_t>(input, "max_frames", 300);
        config.video_input.frame_id = get_or_default<std::string>(input, "frame_id", "camera");
        config.video_input.source_id = get_or_default<std::string>(input, "source_id", "video");

        const YAML::Node model = root["model"];
        config.model_type = get_or_default<std::string>(model, "type", "onnx");
        config.model.model_path = get_or_default<std::string>(model, "path", "models/yolov26.onnx");
        config.model.device = parse_device(model);

        const YAML::Node preprocess = root["preprocess"];
        config.preprocess.model_width = get_or_default<int>(preprocess, "width", 640);
        config.preprocess.model_height = get_or_default<int>(preprocess, "height", 640);
        config.preprocess.normalize = get_or_default<bool>(preprocess, "normalize", true);

        const YAML::Node detection = root["detection"];
        config.detection.confidence_threshold =
            get_or_default<float>(detection, "confidence_threshold", 0.35F);
        config.detection.iou_threshold =
            get_or_default<float>(detection, "iou_threshold", 0.45F);
        config.detection.target_class_id =
            get_or_default<int>(detection, "target_class_id", 15);
        config.detection.target_class_name =
            get_or_default<std::string>(detection, "target_class_name", "cat");

        const YAML::Node target = root["target"];
        config.target_type = get_or_default<std::string>(target, "type", "display");
        config.display_target.window_name =
            get_or_default<std::string>(target, "window_name", "ptk cat detection");
        config.display_target.wait_key_ms =
            get_or_default<int>(target, "wait_key_ms", 1);

        return config;
    }

} // namespace ptk