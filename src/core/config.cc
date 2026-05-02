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
        config.input.width = get_or_default<int>(input, "width", 640);
        config.input.height = get_or_default<int>(input, "height", 480);
        config.input.channels = get_or_default<int>(input, "channels", 3);
        config.input.max_frames = get_or_default<uint64_t>(input, "max_frames", 10);
        config.input.frame_id = get_or_default<std::string>(input, "frame_id", "camera");
        config.input.source_id = get_or_default<std::string>(input, "source_id", "synthetic");

        const YAML::Node model = root["model"];
        config.model.model_path = get_or_default<std::string>(model, "path", "fake");
        config.model.device = parse_device(model);

        const YAML::Node detection = root["detection"];
        config.detection.confidence_threshold =
            get_or_default<float>(detection, "confidence_threshold", 0.5F);
        config.detection.target_class_id =
            get_or_default<int>(detection, "target_class_id", 15);
        config.detection.target_class_name =
            get_or_default<std::string>(detection, "target_class_name", "cat");

        const YAML::Node pipeline = root["pipeline"];
        config.pipeline.max_frames =
            get_or_default<std::size_t>(pipeline, "max_frames", config.input.max_frames);

        return config;
    }

} // namespace ptk