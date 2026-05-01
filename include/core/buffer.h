#pragma once

#include "core/device.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace ptk {

// owns raw memory
class Buffer {
public:
    Buffer() = default;

    Buffer(std::size_t size_bytes, Device device = Device::cpu())
        : size_bytes_(size_bytes), device_(device) {
        if (!device_.is_cpu()) {
            throw std::runtime_error("only cpu buffer allocation is supported right now");
        }

        if (size_bytes_ > 0) {
            data_ = std::make_unique<uint8_t[]>(size_bytes_);
        }
    }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&&) noexcept = default;
    Buffer& operator=(Buffer&&) noexcept = default;

    void* data() {
        return data_.get();
    }

    const void* data() const {
        return data_.get();
    }

    std::size_t size_bytes() const {
        return size_bytes_;
    }

    Device device() const {
        return device_;
    }

    bool empty() const {
        return data_ == nullptr || size_bytes_ == 0;
    }

private:
    std::unique_ptr<uint8_t[]> data_;
    std::size_t size_bytes_ = 0;
    Device device_ = Device::cpu();
};

// non-owning view into raw memory
struct BufferView {
    void* data = nullptr;
    std::size_t size_bytes = 0;
    Device device = Device::cpu();

    bool valid() const {
        return data != nullptr && size_bytes > 0;
    }
};

} // namespace ptk