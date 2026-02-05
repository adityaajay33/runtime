#pragma once

namespace ptk::core {

//output port for writing data to a bound buffer
template <typename T>
class OutputPort {
public:
    OutputPort() : value_(nullptr) {}

    void Bind(T* value) {
        value_ = value;
    }

    bool is_bound() const {
        return value_ != nullptr;
    }

    T* get() const {
        return value_;
    }

private:
    T* value_;
};

//input port for reading data from a bound buffer
template <typename T>
class InputPort {
public:
    InputPort() : value_(nullptr) {}

    void Bind(T* value) {
        value_ = const_cast<T*>(value);
    }

    bool is_bound() const {
        return value_ != nullptr;
    }

    const T* get() const {
        return value_;
    }
    
    //check if data is available (always true if bound for simple port)
    bool HasData() const {
        return value_ != nullptr;
    }
    
    //read reference to the data
    const T& Read() const {
        return *value_;
    }

private:
    T* value_;
};

} //namespace ptk::core