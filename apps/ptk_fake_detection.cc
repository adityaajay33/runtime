#include "models/fake_model_engine.h"
#include "models/model_engine.h"
#include "pipeline/console_target.h"
#include "pipeline/detection_decoder.h"
#include "pipeline/perception_pipeline.h"
#include "pipeline/synthetic_frame_input.h"

#include <iostream>

int main()
{
    ptk::SyntheticFrameInputConfig input_config;
    input_config.width = 640;
    input_config.height = 480;
    input_config.channels = 3;
    input_config.max_frames = 10;
    input_config.frame_id = "camera";
    input_config.source_id = "synthetic";

    ptk::SyntheticFrameInput input(input_config);

    ptk::FakeModelEngine model;
    ptk::ModelConfig model_config;
    model_config.model_path = "fake";

    if (!model.load_model(model_config))
    {
        std::cerr << "failed to load fake model\n";
        return 1;
    }

    ptk::DetectionDecoderConfig decoder_config;
    decoder_config.confidence_threshold = 0.5F;
    decoder_config.target_class_id = 15;
    decoder_config.target_class_name = "cat";

    ptk::DetectionDecoder decoder(decoder_config);
    ptk::ConsoleTarget target;

    ptk::PerceptionPipelineConfig pipeline_config;
    pipeline_config.max_frames = input_config.max_frames;

    ptk::PerceptionPipeline pipeline(
        input,
        model,
        decoder,
        target,
        pipeline_config);

    if (!pipeline.run())
    {
        std::cerr << "pipeline failed\n";
        return 1;
    }

    return 0;
}