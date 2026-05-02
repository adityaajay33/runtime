#pragma once

#include "models/model_engine.h"

namespace ptk {

class FakeModelEngine : public ModelEngine {
public:
    FakeModelEngine() = default;

    bool load_model(const ModelConfig& config) override;
    bool run(const TensorView& input, TensorView& output) override;

private:
    bool loaded_ = false;
};

} // namespace ptk