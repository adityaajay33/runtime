#pragma once

#include "models/model_engine.h"

#include <onnxruntime_cxx_api.h>

#include <memory>
#include <string>
#include <vector>

namespace ptk
{

    class OnnxModelEngine : public ModelEngine
    {
    public:
        OnnxModelEngine();

        bool load_model(const ModelConfig &config) override;
        bool run(const TensorView &input, TensorView &output) override;

        TensorShape input_shape() const override;
        TensorShape output_shape() const override;

        const std::vector<std::string> &input_names() const;
        const std::vector<std::string> &output_names() const;

    private:
        bool read_model_metadata();
        TensorShape read_shape(std::vector<int64_t> dims) const;
        DataType read_dtype(ONNXTensorElementDataType dtype) const;

        Ort::Env env_;
        Ort::SessionOptions session_options_;
        std::unique_ptr<Ort::Session> session_;

        ModelConfig config_;
        bool loaded_ = false;

        std::vector<std::string> input_names_;
        std::vector<std::string> output_names_;

        TensorShape input_shape_;
        TensorShape output_shape_;
        DataType input_dtype_ = DataType::Float32;
        DataType output_dtype_ = DataType::Float32;
    };

} // namespace ptk