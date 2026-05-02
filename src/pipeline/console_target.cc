#include "pipeline/console_target.h"

#include <iomanip>
#include <iostream>

namespace ptk
{

    bool ConsoleTarget::open()
    {
        frames_written_ = 0;
        opened_ = true;
        return true;
    }

    bool ConsoleTarget::write(const PerceptionResult &result, const Frame &frame)
    {
        if (!opened_)
        {
            return false;
        }

        std::cout << "frame=" << frame.sequence
                  << " source=" << frame.source_id
                  << " size=" << frame.width << "x" << frame.height
                  << " detections=" << result.detections.size()
                  << '\n';

        for (const auto &detection : result.detections)
        {
            std::cout << "  class=" << detection.class_name
                      << " id=" << detection.class_id
                      << " confidence=" << std::fixed << std::setprecision(3)
                      << detection.confidence
                      << " bbox=("
                      << detection.bbox.x << ", "
                      << detection.bbox.y << ", "
                      << detection.bbox.width << ", "
                      << detection.bbox.height << ")"
                      << '\n';
        }

        ++frames_written_;
        return true;
    }

    void ConsoleTarget::close()
    {
        opened_ = false;
    }

} // namespace ptk