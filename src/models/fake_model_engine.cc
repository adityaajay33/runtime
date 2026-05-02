#include "models/fake_model_engine.h"

namespace ptk
{

    bool FakeModelEngine::load_model(const ModelConfig & /*config*/)
    {
        loaded_ = true;
        return true;
    }

    bool FakeModelEngine::run(const TensorView &input, TensorView &output)
    {
        if (!loaded_ || !input.valid() || !output.valid())
        {
            return false;
        }

        auto *data = output.data_as<float>();

        if (output.num_elements() < 6)
        {
            return false;
        }

        // fake detection format: x, y, width, height, confidence, class_id
        data[0] = 100.0F;
        data[1] = 80.0F;
        data[2] = 220.0F;
        data[3] = 160.0F;
        data[4] = 0.92F;
        data[5] = 15.0F;

        return true;
    }

} // namespace ptk