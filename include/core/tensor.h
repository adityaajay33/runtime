#pragma once

#include "core/buffer.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ptk {

enum class DataType {
    UInt8,
    Float32,
    Float16,
    Int32,
    Int64
};

inline std::size_t byte_size(DataType dtype) {
    switch (dtype) {
        case DataType::UInt8:
            return 1;
        case DataType::Float32:
            return 4;
        case DataType::Float16:
            return 2;
        case DataType::Int32:
            return 4;
        case DataType::Int64:
            return 8;
        default:
            throw std::runtime_error("unsupported data type");
    }
}

enum class TensorLayout {
    Unknown,
    HWC,
    CHW,
    NCHW,
    NHWC,
    Flat
};

struct TensorShape {
    std::vector<int64_t> dims;

    TensorShape() = default;

    explicit TensorShape(std::vector<int64_t> dimensions)
        : dims(std::move(dimensions)) {}

    std::size_t rank() const {
        return dims.size();
    }

    std::size_t num_elements() const {
        if (dims.empty()) {
            return 0;
        }

        std::size_t total = 1;

        for (int64_t dim : dims) {
            if (dim <= 0) {
                throw std::runtime_error("tensor shape contains non-positive dimension");
            }

            total *= static_cast<std::size_t>(dim);
        }

        return total;
    }

    std::size_t size_bytes(DataType dtype) const {
        return num_elements() * byte_size(dtype);
    }
};

// non-owning view into tensor memory
struct TensorView {
    BufferView buffer;
    TensorShape shape;
    DataType dtype = DataType::Float32;
    TensorLayout layout = TensorLayout::Unknown;

    std::size_t num_elements() const {
        return shape.num_elements();
    }

    std::size_t expected_size_bytes() const {
        return shape.size_bytes(dtype);
    }

    bool valid() const {
        return buffer.valid() && buffer.size_bytes >= expected_size_bytes();
    }

    template <typename T>
    T* data_as() const {
        return static_cast<T*>(buffer.data);
    }
};

// owns tensor memory
class Tensor {
public:
    Tensor() = default;

    Tensor(
        TensorShape shape,
        DataType dtype,
        TensorLayout layout,
        Device device = Device::cpu()
    )
        : shape_(std::move(shape)),
          dtype_(dtype),
          layout_(layout),
          buffer_(shape_.size_bytes(dtype_), device) {}

    TensorView view() {
        return TensorView{
            BufferView{buffer_.data(), buffer_.size_bytes(), buffer_.device()},
            shape_,
            dtype_,
            layout_
        };
    }

    const TensorShape& shape() const {
        return shape_;
    }

    DataType dtype() const {
        return dtype_;
    }

    TensorLayout layout() const {
        return layout_;
    }

    Device device() const {
        return buffer_.device();
    }

    std::size_t size_bytes() const {
        return buffer_.size_bytes();
    }

private:
    TensorShape shape_;
    DataType dtype_ = DataType::Float32;
    TensorLayout layout_ = TensorLayout::Unknown;
    Buffer buffer_;
};

} // namespace ptk