#pragma once

#include "runtime/components/component_interface.h"
#include "runtime/core/port.h"
#include "runtime/data/frame.h"
#include "tasks/task_output.h"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

namespace ptk::components {

class FramePublisherBridge : public ComponentInterface {
public:
    explicit FramePublisherBridge(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
    ~FramePublisherBridge() override = default;

    void BindInput(core::InputPort<data::Frame>* in);

    core::Status Init(core::RuntimeContext* context) override;
    core::Status Start() override;
    core::Status Stop() override;
    void Tick() override;

private:
    core::RuntimeContext* context_;
    core::InputPort<data::Frame>* input_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
    std::string topic_name_;
    std::string frame_id_;
};

class FrameSubscriberBridge : public ComponentInterface {
public:
    explicit FrameSubscriberBridge(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
    ~FrameSubscriberBridge() override = default;

    void BindOutput(core::OutputPort<data::Frame>* out);

    core::Status Init(core::RuntimeContext* context) override;
    core::Status Start() override;
    core::Status Stop() override;
    void Tick() override;

private:
    void ImageCallback(const sensor_msgs::msg::Image::SharedPtr msg);

    core::RuntimeContext* context_;
    core::OutputPort<data::Frame>* output_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    std::string topic_name_;
    
    std::vector<uint8_t> frame_buffer_;
    bool has_new_frame_;
    std::mutex mutex_;
};

} //namespace ptk::components
