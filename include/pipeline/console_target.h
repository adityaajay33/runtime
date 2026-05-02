#pragma once

#include "core/detection.h"
#include "pipeline/perception_target.h"

#include <cstddef>

namespace ptk
{

    class ConsoleTarget : public PerceptionTarget
    {
    public:
        ConsoleTarget() = default;

        bool open() override;
        bool write(const PerceptionResult &result, const Frame &frame) override;
        void close() override;

    private:
        std::size_t frames_written_ = 0;
        bool opened_ = false;
    };

} // namespace ptk