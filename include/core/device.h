#pragma once

namespace ptk
{

    enum class DeviceType
    {
        Cpu,
        Cuda,
        Mps
    };

    struct Device
    {
        DeviceType type = DeviceType::Cpu;
        int index = 0;

        static Device cpu()
        {
            return Device{DeviceType::Cpu, 0};
        }

        static Device cuda(int device_index = 0)
        {
            return Device{DeviceType::Cuda, device_index};
        }

        static Device mps()
        {
            return Device{DeviceType::Mps, 0};
        }

        bool is_cpu() const
        {
            return type == DeviceType::Cpu;
        }

        bool is_cuda() const
        {
            return type == DeviceType::Cuda;
        }

        bool is_mps() const
        {
            return type == DeviceType::Mps;
        }
    };

} // namespace ptk