#include "pipeline/perception_pipeline.h"

#include <iostream>

namespace ptk
{

    PerceptionPipeline::PerceptionPipeline(
        FrameInput &input,
        ModelEngine &model,
        DetectionDecoder &decoder,
        PerceptionTarget &target,
        PerceptionPipelineConfig config)
        : input_(input),
          model_(model),
          decoder_(decoder),
          target_(target),
          config_(config) {}

    bool PerceptionPipeline::run()
    {
        if (!input_.open())
        {
            std::cerr << "failed to open frame input\n";
            return false;
        }

        if (!target_.open())
        {
            std::cerr << "failed to open perception target\n";
            input_.close();
            return false;
        }

        Tensor model_output(
            TensorShape({6}),
            DataType::Float32,
            TensorLayout::Flat,
            Device::cpu());

        TensorView model_output_view = model_output.view();

        Frame frame;
        PerceptionResult result;

        std::size_t frames_processed = 0;

        while (frames_processed < config_.max_frames && input_.read_next(frame))
        {
            if (!model_.run(frame.image, model_output_view))
            {
                std::cerr << "model run failed on frame " << frame.sequence << '\n';
                target_.close();
                input_.close();
                return false;
            }

            if (!decoder_.run(model_output_view, frame, result))
            {
                std::cerr << "detection decode failed on frame " << frame.sequence << '\n';
                target_.close();
                input_.close();
                return false;
            }

            if (!target_.write(result, frame))
            {
                std::cerr << "target write failed on frame " << frame.sequence << '\n';
                target_.close();
                input_.close();
                return false;
            }

            ++frames_processed;
        }

        target_.close();
        input_.close();

        return true;
    }

} // namespace ptk