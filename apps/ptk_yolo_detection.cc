#include "core/config.h"
#include "core/tensor.h"
#include "models/onnx_model_engine.h"
#include "pipeline/display_target.h"
#include "pipeline/image_preprocessor.h"
#include "pipeline/video_frame_input.h"
#include "pipeline/yolo_detection_decoder.h"

#include <exception>
#include <iostream>

int main()
{
    try
    {
        const ptk::AppConfig config = ptk::load_config("config.yaml");

        ptk::VideoFrameInput input(config.video_input);
        ptk::ImagePreprocessor preprocessor(config.preprocess);
        ptk::OnnxModelEngine model;
        ptk::YoloDetectionDecoder decoder(config.detection);
        ptk::DisplayTarget target(config.display_target);

        if (!input.open())
        {
            std::cerr << "[ptk] failed to open video: " << config.video_input.path << '\n';
            return 1;
        }
        std::cout << "[ptk] video opened: " << config.video_input.path << '\n';

        std::cout << "[ptk] loading model: " << config.model.model_path << '\n';
        if (!model.load_model(config.model))
        {
            std::cerr << "[ptk] failed to load ONNX model: " << config.model.model_path << '\n';
            input.close();
            return 1;
        }
        std::cout << "[ptk] model loaded\n";

        if (!target.open())
        {
            std::cerr << "[ptk] failed to open display target\n";
            input.close();
            return 1;
        }
        std::cout << "[ptk] display ready: " << config.display_target.window_name << '\n';

        ptk::Tensor model_input(
            ptk::TensorShape({1,
                              3,
                              static_cast<int64_t>(config.preprocess.model_height),
                              static_cast<int64_t>(config.preprocess.model_width)}),
            ptk::DataType::Float32,
            ptk::TensorLayout::NCHW,
            ptk::Device::cpu());

        ptk::Tensor model_output(
            model.output_shape(),
            model.output_dtype(),
            model.output_layout(),
            ptk::Device::cpu());

        ptk::TensorView model_input_view = model_input.view();
        ptk::TensorView model_output_view = model_output.view();

        ptk::Frame frame;
        ptk::PreprocessInfo preprocess_info;
        ptk::PerceptionResult result;

        while (input.read_next(frame))
        {
            if (!preprocessor.run(frame, model_input_view, preprocess_info))
            {
                std::cerr << "preprocess failed on frame " << frame.sequence << '\n';
                break;
            }

            if (!model.run(model_input_view, model_output_view))
            {
                std::cerr << "model inference failed on frame " << frame.sequence << '\n';
                break;
            }

            if (!decoder.run(model_output_view, frame, preprocess_info, result))
            {
                std::cerr << "YOLO decode failed on frame " << frame.sequence << '\n';
                break;
            }

            if (!target.write(result, frame))
            {
                break;
            }
        }

        target.close();
        input.close();

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}