#include "runtime/components/topic_bridge.h"
#include "runtime/core/runtime_context.h"
#include <rclcpp_components/register_node_macro.hpp>

namespace ptk::components {

FramePublisherBridge::FramePublisherBridge(const rclcpp::NodeOptions& options)
    : ComponentInterface("frame_publisher_bridge", options),
      context_(nullptr),
      input_(nullptr),
      topic_name_("ptk/frame"),
      frame_id_("camera")
{
    this->declare_parameter("topic_name", "ptk/frame");
    this->declare_parameter("frame_id", "camera");
    topic_name_ = this->get_parameter("topic_name").as_string();
    frame_id_ = this->get_parameter("frame_id").as_string();
}

void FramePublisherBridge::BindInput(core::InputPort<data::Frame>* in) {
    input_ = in;
}

core::Status FramePublisherBridge::Init(core::RuntimeContext* context) {
    context_ = context;
    
    //create publisher with qos for real-time imaging
    auto qos = rclcpp::QoS(10).best_effort();
    pub_ = this->create_publisher<sensor_msgs::msg::Image>(topic_name_, qos);
    
    RCLCPP_INFO(this->get_logger(), "initialized frame publisher on topic: %s", topic_name_.c_str());
    return core::Status::Ok();
}

core::Status FramePublisherBridge::Start() {
    return core::Status::Ok();
}

core::Status FramePublisherBridge::Stop() {
    return core::Status::Ok();
}

void FramePublisherBridge::Tick() {
    if (!input_ || !input_->is_bound()) {
        return;
    }
    
    const data::Frame* frame = input_->get();
    if (!frame || frame->image.empty()) {
        return;
    }
    
    auto msg = sensor_msgs::msg::Image();
    msg.header.stamp = this->get_clock()->now();
    msg.header.frame_id = frame_id_;
    
    const auto& shape = frame->image.shape();
    msg.height = static_cast<uint32_t>(shape.dim(0));
    msg.width = static_cast<uint32_t>(shape.dim(1));
    
    if (frame->pixel_format == core::PixelFormat::kRgb8) {
        msg.encoding = "rgb8";
    } else if (frame->pixel_format == core::PixelFormat::kBgr8) {
        msg.encoding = "bgr8";
    } else if (frame->pixel_format == core::PixelFormat::kGray8) {
        msg.encoding = "mono8";
    } else {
        msg.encoding = "rgb8";
    }
    
    int channels = (frame->pixel_format == core::PixelFormat::kGray8) ? 1 : 3;
    msg.step = msg.width * channels;
    
    const uint8_t* src = static_cast<const uint8_t*>(frame->image.buffer().data());
    size_t data_size = msg.height * msg.step;
    msg.data.assign(src, src + data_size);
    
    pub_->publish(msg);
}

//frame subscriber bridge implementation
FrameSubscriberBridge::FrameSubscriberBridge(const rclcpp::NodeOptions& options)
    : ComponentInterface("frame_subscriber_bridge", options),
      context_(nullptr),
      output_(nullptr),
      topic_name_("camera/image_raw"),
      has_new_frame_(false)
{
    this->declare_parameter("topic_name", "camera/image_raw");
    topic_name_ = this->get_parameter("topic_name").as_string();
}

void FrameSubscriberBridge::BindOutput(core::OutputPort<data::Frame>* out) {
    output_ = out;
}

core::Status FrameSubscriberBridge::Init(core::RuntimeContext* context) {
    context_ = context;
    
    //create subscription with qos for real-time imaging
    auto qos = rclcpp::QoS(10).best_effort();
    sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        topic_name_, qos,
        std::bind(&FrameSubscriberBridge::ImageCallback, this, std::placeholders::_1));
    
    RCLCPP_INFO(this->get_logger(), "initialized frame subscriber on topic: %s", topic_name_.c_str());
    return core::Status::Ok();
}

core::Status FrameSubscriberBridge::Start() {
    return core::Status::Ok();
}

core::Status FrameSubscriberBridge::Stop() {
    return core::Status::Ok();
}

void FrameSubscriberBridge::ImageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!output_ || !output_->is_bound()) {
        return;
    }
    
    //copy image data to internal buffer
    frame_buffer_.assign(msg->data.begin(), msg->data.end());
    
    data::Frame* out = output_->get();
    if (!out) {
        return;
    }
    
    int H = static_cast<int>(msg->height);
    int W = static_cast<int>(msg->width);
    int C = (msg->encoding == "mono8") ? 1 : 3;
    
    out->image = data::TensorView(
        data::BufferView(frame_buffer_.data(), frame_buffer_.size(), core::DeviceType::kCpu),
        core::DataType::kUint8,
        data::TensorShape({H, W, C})
    );
    
    //set pixel format from encoding
    if (msg->encoding == "rgb8") {
        out->pixel_format = core::PixelFormat::kRgb8;
    } else if (msg->encoding == "bgr8") {
        out->pixel_format = core::PixelFormat::kBgr8;
    } else if (msg->encoding == "mono8") {
        out->pixel_format = core::PixelFormat::kGray8;
    } else {
        out->pixel_format = core::PixelFormat::kRgb8;
    }
    
    out->layout = core::TensorLayout::kHwc;
    out->timestamp_ns = msg->header.stamp.nanosec + msg->header.stamp.sec * 1000000000LL;
    out->frame_index = 0;
    out->camera_id = 0;
    
    has_new_frame_ = true;
}

void FrameSubscriberBridge::Tick() {
    //ros subscription callback handles data - tick is a no-op for subscriber
}

} //namespace ptk::components

RCLCPP_COMPONENTS_REGISTER_NODE(ptk::components::FramePublisherBridge)
RCLCPP_COMPONENTS_REGISTER_NODE(ptk::components::FrameSubscriberBridge)
