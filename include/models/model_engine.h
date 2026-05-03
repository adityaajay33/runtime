#pragma once

#include "core/device.h"
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

        virtual TensorShape input_shape() const = 0;
        virtual TensorShape output_shape() const = 0;

        virtual DataType input_dtype() const
        {
            return DataType::Float32;
        }

        virtual DataType output_dtype() const
        {
            return DataType::Float32;
        }

        virtual TensorLayout input_layout() const
        {
            return TensorLayout::NCHW;
        }

        virtual TensorLayout output_layout() const
        {
            return TensorLayout::Flat;
        }
    };

} // namespace ptk