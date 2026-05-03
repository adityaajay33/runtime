#include "models/onnx_model_engine.h"

#include <iostream>
#include <string>

namespace
{

    void print_shape(const ptk::TensorShape &shape)
    {
        std::cout << "[";

        for (std::size_t i = 0; i < shape.dims.size(); ++i)
        {
            std::cout << shape.dims[i];

            if (i + 1 < shape.dims.size())
            {
                std::cout << ", ";
            }
        }

        std::cout << "]";
    }

} // namespace

int main(int argc, char **argv)
{
    std::string model_path = "models/yolo26.onnx";

    if (argc > 1)
    {
        model_path = argv[1];
    }

    ptk::ModelConfig config;
    config.model_path = model_path;
    config.device = ptk::Device::cpu();

    ptk::OnnxModelEngine engine;

    if (!engine.load_model(config))
    {
        std::cerr << "failed to load ONNX model: " << model_path << '\n';
        return 1;
    }

    std::cout << "inspection complete\n";

    std::cout << "input shape: ";
    print_shape(engine.input_shape());
    std::cout << '\n';

    std::cout << "output shape: ";
    print_shape(engine.output_shape());
    std::cout << '\n';

    return 0;
}