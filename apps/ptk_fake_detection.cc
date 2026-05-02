#include "core/config.h"
#include "models/fake_model_engine.h"
#include "pipeline/console_target.h"
#include "pipeline/detection_decoder.h"
#include "pipeline/perception_pipeline.h"
#include "pipeline/synthetic_frame_input.h"

#include <exception>
#include <iostream>

int main() {
    try {
        const ptk::AppConfig config = ptk::load_config("config.yaml");

        ptk::SyntheticFrameInput input(config.input);

        ptk::FakeModelEngine model;
        if (!model.load_model(config.model)) {
            std::cerr << "failed to load fake model\n";
            return 1;
        }

        ptk::DetectionDecoder decoder(config.detection);
        ptk::ConsoleTarget target;

        ptk::PerceptionPipeline pipeline(
            input,
            model,
            decoder,
            target,
            config.pipeline
        );

        if (!pipeline.run()) {
            std::cerr << "pipeline failed\n";
            return 1;
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}