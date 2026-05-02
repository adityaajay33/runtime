#pragma once

#include "core/tensor.h"

#include <string>

namespace ptk
{
    struct ModelConfig
    {
        std::string model_path;
        Device device = Device::cpu();
    };

    class ModelEngine
    {
        public:
            virtual ~ModelEngine() = default;

            virtual bool load_model(const ModelConfig &config) = 0;
            virtual bool run(const TensorView &input, TensorView &output) = 0;
    };
} // namespace ptk