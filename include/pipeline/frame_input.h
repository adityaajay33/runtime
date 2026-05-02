#pragma once

#include "core/frame.h"

namespace ptk
{
    class FrameInput
    {
    public:
        virtual ~FrameInput() = default;

        virtual bool open() = 0;
        virtual bool read_next(Frame &frame) = 0;
        virtual void close() = 0;
    };
} // namespace ptk