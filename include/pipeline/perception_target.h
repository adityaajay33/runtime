#pragma once

#include "core/detection.h"

namespace ptk
{

    class PerceptionTarget
    {
    public:
        virtual ~PerceptionTarget() = default;

        virtual bool open() = 0;
        virtual bool write(const PerceptionResult &result, const Frame &frame) = 0;
        virtual void close() = 0;
    };

} // namespace ptk