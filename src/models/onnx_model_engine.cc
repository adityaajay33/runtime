#include "models/onnx_model_engine.h"

#include <iostream>
#include <stdexcept>
#include <utility>

namespace ptk
{

    namespace
    {

        ONNXTensorElementDataType to_onnx_dtype(DataType dtype)
        {
            switch (dtype)
            {
            case DataType::UInt8:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
            case DataType::Float32:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
            case DataType::Float16:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
            case DataType::Int32:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
            case DataType::Int64:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
            default:
                throw std::runtime_error("unsupported PTK data type for ONNX Runtime");
            }
        }

    } // namespace

    OnnxModelEngine::OnnxModelEngine()
        : env_(ORT_LOGGING_LEVEL_WARNING, "ptk-onnx")
    {
        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    }

    bool OnnxModelEngine::load_model(const ModelConfig &config)
    {
        try
        {
            config_ = config;

            if (!config_.device.is_cpu())
            {
                std::cerr << "OnnxModelEngine currently supports CPU execution only\n";
                return false;
            }

            session_ = std::make_unique<Ort::Session>(
                env_,
                config_.model_path.c_str(),
                session_options_);

            if (!read_model_metadata())
            {
                return false;
            }

            loaded_ = true;
            return true;
        }
        catch (const Ort::Exception &error)
        {
            std::cerr << "ONNX Runtime error while loading model: "
                      << error.what() << '\n';
            return false;
        }
        catch (const std::exception &error)
        {
            std::cerr << "error while loading ONNX model: "
                      << error.what() << '\n';
            return false;
        }
    }

    bool OnnxModelEngine::run(const TensorView &input, TensorView &output)
    {
        if (!loaded_ || !session_ || !input.valid() || !output.valid())
        {
            return false;
        }

        if (input.dtype != input_dtype_ || output.dtype != output_dtype_)
        {
            return false;
        }

        try
        {
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator,
                OrtMemTypeDefault);

            auto input_shape_dims = input.shape.dims;
            auto output_shape_dims = output.shape.dims;

            Ort::Value input_tensor = Ort::Value::CreateTensor(
                memory_info,
                input.buffer.data,
                input.buffer.size_bytes,
                input_shape_dims.data(),
                input_shape_dims.size(),
                to_onnx_dtype(input.dtype));

            Ort::Value output_tensor = Ort::Value::CreateTensor(
                memory_info,
                output.buffer.data,
                output.buffer.size_bytes,
                output_shape_dims.data(),
                output_shape_dims.size(),
                to_onnx_dtype(output.dtype));

            const char *input_name = input_names_.front().c_str();
            const char *output_name = output_names_.front().c_str();

            session_->Run(
                Ort::RunOptions{nullptr},
                &input_name,
                &input_tensor,
                1,
                &output_name,
                &output_tensor,
                1);

            return true;
        }
        catch (const Ort::Exception &error)
        {
            std::cerr << "ONNX Runtime error while running inference: "
                      << error.what() << '\n';
            return false;
        }
        catch (const std::exception &error)
        {
            std::cerr << "error while running ONNX inference: "
                      << error.what() << '\n';
            return false;
        }
    }

    TensorShape OnnxModelEngine::input_shape() const
    {
        return input_shape_;
    }

    TensorShape OnnxModelEngine::output_shape() const
    {
        return output_shape_;
    }

    const std::vector<std::string> &OnnxModelEngine::input_names() const
    {
        return input_names_;
    }

    const std::vector<std::string> &OnnxModelEngine::output_names() const
    {
        return output_names_;
    }

    bool OnnxModelEngine::read_model_metadata()
    {
        if (!session_)
        {
            return false;
        }

        Ort::AllocatorWithDefaultOptions allocator;

        const std::size_t input_count = session_->GetInputCount();
        const std::size_t output_count = session_->GetOutputCount();

        if (input_count != 1 || output_count != 1)
        {
            std::cerr << "OnnxModelEngine currently supports one input and one output only\n";
            return false;
        }

        input_names_.clear();
        output_names_.clear();

        auto input_name = session_->GetInputNameAllocated(0, allocator);
        auto output_name = session_->GetOutputNameAllocated(0, allocator);

        input_names_.emplace_back(input_name.get());
        output_names_.emplace_back(output_name.get());

        const auto input_type_info = session_->GetInputTypeInfo(0);
        const auto output_type_info = session_->GetOutputTypeInfo(0);

        const auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        const auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();

        input_shape_ = read_shape(input_tensor_info.GetShape());
        output_shape_ = read_shape(output_tensor_info.GetShape());

        input_dtype_ = read_dtype(input_tensor_info.GetElementType());
        output_dtype_ = read_dtype(output_tensor_info.GetElementType());

        std::cout << "loaded ONNX model: " << config_.model_path << '\n';
        std::cout << "  input:  " << input_names_.front() << " shape=[";
        for (std::size_t i = 0; i < input_shape_.dims.size(); ++i)
        {
            std::cout << input_shape_.dims[i];
            if (i + 1 < input_shape_.dims.size())
            {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";

        std::cout << "  output: " << output_names_.front() << " shape=[";
        for (std::size_t i = 0; i < output_shape_.dims.size(); ++i)
        {
            std::cout << output_shape_.dims[i];
            if (i + 1 < output_shape_.dims.size())
            {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";

        return true;
    }

    TensorShape OnnxModelEngine::read_shape(std::vector<int64_t> dims) const
    {
        for (auto &dim : dims)
        {
            if (dim < 0)
            {
                dim = 1;
            }
        }

        return TensorShape(std::move(dims));
    }

    DataType OnnxModelEngine::read_dtype(ONNXTensorElementDataType dtype) const
    {
        switch (dtype)
        {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return DataType::UInt8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return DataType::Float32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return DataType::Float16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return DataType::Int32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return DataType::Int64;
        default:
            throw std::runtime_error("unsupported ONNX tensor element type");
        }
    }

} // namespace ptk