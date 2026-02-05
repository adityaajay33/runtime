#include <rclcpp/rclcpp.hpp>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <csignal>

#include "sensors/mac_camera.h"
#include "operators/preprocessor.h"
#include "runtime/components/inference_node.h"
#include "runtime/core/runtime_context.h"
#include "runtime/core/scheduler.h"
#include "runtime/core/port.h"
#include "runtime/data/frame.h"
#include "tasks/task_output.h"

using namespace ptk;

namespace {
    volatile std::sig_atomic_t g_running = 1;
    
    void SignalHandler(int signal) {
        (void)signal;
        g_running = 0;
    }
}

class DetectionLogger : public components::ComponentInterface {
public:
    explicit DetectionLogger(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : ComponentInterface("detection_logger", options),
          context_(nullptr),
          input_(nullptr),
          frames_processed_(0)
    {
    }

    void BindInput(core::InputPort<tasks::TaskOutput>* in) {
        input_ = in;
    }

    core::Status Init(core::RuntimeContext* context) override {
        context_ = context;
        RCLCPP_INFO(this->get_logger(), "detection logger initialized");
        return core::Status::Ok();
    }

    core::Status Start() override {
        frames_processed_ = 0;
        return core::Status::Ok();
    }

    core::Status Stop() override {
        RCLCPP_INFO(this->get_logger(), "processed %d frames total", frames_processed_);
        return core::Status::Ok();
    }

    void Tick() override {
        if (!input_ || !input_->is_bound()) {
            return;
        }

        const tasks::TaskOutput* result = input_->get();
        if (!result || !result->success) {
            return;
        }

        frames_processed_++;
        
        //log detection summary
        if (!result->detections.empty()) {
            std::cout << "[Frame " << result->frame_index << "] "
                      << result->detections.size() << " detections, "
                      << result->inference_time_ms << "ms" << std::endl;
            
            for (const auto& det : result->detections) {
                std::cout << "  - " << det.class_name 
                          << " (conf=" << det.confidence << ") "
                          << "[" << det.box.x1 << "," << det.box.y1 
                          << "," << det.box.x2 << "," << det.box.y2 << "]" 
                          << std::endl;
            }
        } else {
            //only log every 10th frame if no detections
            if (frames_processed_ % 10 == 0) {
                std::cout << "[Frame " << result->frame_index << "] "
                          << "no detections, " << result->inference_time_ms << "ms" 
                          << std::endl;
            }
        }
    }

private:
    core::RuntimeContext* context_;
    core::InputPort<tasks::TaskOutput>* input_;
    int frames_processed_;
};

void PrintUsage(const char* program) {
    std::cout << "Usage: " << program << " <model_path> [options]\n"
              << "\nOptions:\n"
              << "  --device <index>      Camera device index (default: 0)\n"
              << "  --width <pixels>      Preprocessing width (default: 224)\n"
              << "  --height <pixels>     Preprocessing height (default: 224)\n"
              << "  --confidence <float>  Detection confidence threshold (default: 0.5)\n"
              << "  --fps <hz>            Target framerate (default: 30)\n"
              << "  --frames <count>      Number of frames to process (-1 for infinite, default: -1)\n"
              << "  --help                Show this help message\n"
              << "\nExample:\n"
              << "  " << program << " /path/to/model.onnx --device 0 --width 640 --height 480\n"
              << std::endl;
}

int main(int argc, char** argv) {
    //check for help or missing args
    if (argc < 2) {
        PrintUsage(argv[0]);
        return 1;
    }
    
    std::string model_path = argv[1];
    if (model_path == "--help" || model_path == "-h") {
        PrintUsage(argv[0]);
        return 0;
    }
    
    int device_index = 0;
    int target_width = 224;
    int target_height = 224;
    double confidence_threshold = 0.5;
    int fps = 30;
    int num_frames = -1;
    
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) {
            device_index = std::stoi(argv[++i]);
        } else if (arg == "--width" && i + 1 < argc) {
            target_width = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            target_height = std::stoi(argv[++i]);
        } else if (arg == "--confidence" && i + 1 < argc) {
            confidence_threshold = std::stod(argv[++i]);
        } else if (arg == "--fps" && i + 1 < argc) {
            fps = std::stoi(argv[++i]);
        } else if (arg == "--frames" && i + 1 < argc) {
            num_frames = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            return 0;
        }
    }
    
    //register signal handler for clean shutdown
    std::signal(SIGINT, SignalHandler);
    std::signal(SIGTERM, SignalHandler);
    
    rclcpp::init(argc, argv);
    
    std::cout << "=== PTK Model Runner ===" << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Device: " << device_index << std::endl;
    std::cout << "Input size: " << target_width << "x" << target_height << std::endl;
    std::cout << "Confidence: " << confidence_threshold << std::endl;
    std::cout << "FPS: " << fps << std::endl;
    std::cout << std::endl;
    
    core::RuntimeContextOptions ctx_opts;
    ctx_opts.info_stream = stdout;
    ctx_opts.error_stream = stderr;
    
    core::RuntimeContext context;
    auto status = context.Init(ctx_opts);
    if (!status.ok()) {
        std::cerr << "failed to init context: " << status.message() << std::endl;
        return 1;
    }
    
    //create components with parameters
    rclcpp::NodeOptions cam_opts;
    cam_opts.append_parameter_override("device_index", device_index);
    auto camera = std::make_shared<sensors::MacCamera>(cam_opts);
    
    rclcpp::NodeOptions preproc_opts;
    preproc_opts.append_parameter_override("target_width", target_width);
    preproc_opts.append_parameter_override("target_height", target_height);
    preproc_opts.append_parameter_override("normalize", true);
    auto preprocessor = std::make_shared<Preprocessor>(preproc_opts);
    
    rclcpp::NodeOptions inf_opts;
    inf_opts.append_parameter_override("model_path", model_path);
    inf_opts.append_parameter_override("backend", "onnx");
    inf_opts.append_parameter_override("task_type", "detection");
    inf_opts.append_parameter_override("confidence_threshold", confidence_threshold);
    auto inference = std::make_shared<components::InferenceNode>(inf_opts);
    
    auto logger = std::make_shared<DetectionLogger>(rclcpp::NodeOptions());
    
    //create data buffers and ports
    data::Frame camera_frame;
    data::Frame preprocessed_frame;
    tasks::TaskOutput inference_result;
    
    core::OutputPort<data::Frame> camera_out;
    camera_out.Bind(&camera_frame);
    
    core::InputPort<data::Frame> preproc_in;
    preproc_in.Bind(&camera_frame);
    
    core::OutputPort<data::Frame> preproc_out;
    preproc_out.Bind(&preprocessed_frame);
    
    core::InputPort<data::Frame> inference_in;
    inference_in.Bind(&preprocessed_frame);
    
    core::OutputPort<tasks::TaskOutput> inference_out;
    inference_out.Bind(&inference_result);
    
    core::InputPort<tasks::TaskOutput> logger_in;
    logger_in.Bind(&inference_result);
    
    //bind ports to components
    camera->BindOutput(&camera_out);
    preprocessor->BindInput(&preproc_in);
    preprocessor->BindOutput(&preproc_out);
    inference->BindInput(&inference_in);
    inference->BindOutput(&inference_out);
    logger->BindInput(&logger_in);
    
    //initialize camera separately (different interface)
    status = camera->Init();
    if (!status.ok()) {
        std::cerr << "failed to init camera: " << status.message() << std::endl;
        rclcpp::shutdown();
        return 1;
    }
    
    status = camera->Start();
    if (!status.ok()) {
        std::cerr << "failed to start camera: " << status.message() << std::endl;
        rclcpp::shutdown();
        return 1;
    }
    
    //initialize other components
    status = preprocessor->Init(&context);
    if (!status.ok()) {
        std::cerr << "failed to init preprocessor: " << status.message() << std::endl;
        camera->Stop();
        rclcpp::shutdown();
        return 1;
    }
    
    status = inference->Init(&context);
    if (!status.ok()) {
        std::cerr << "failed to init inference: " << status.message() << std::endl;
        camera->Stop();
        rclcpp::shutdown();
        return 1;
    }
    
    status = logger->Init(&context);
    if (!status.ok()) {
        std::cerr << "failed to init logger: " << status.message() << std::endl;
        camera->Stop();
        rclcpp::shutdown();
        return 1;
    }
    
    preprocessor->Start();
    inference->Start();
    logger->Start();
    
    std::cout << "pipeline started, press Ctrl+C to stop" << std::endl;
    std::cout << std::endl;
    
    //calculate frame interval
    auto frame_interval = std::chrono::milliseconds(1000 / fps);
    int frame_count = 0;
    
    //main loop
    while (g_running && (num_frames < 0 || frame_count < num_frames)) {
        auto start = std::chrono::steady_clock::now();
        
        //tick all components in pipeline order
        camera->Tick();
        preprocessor->Tick();
        inference->Tick();
        logger->Tick();
        
        frame_count++;
        

        auto elapsed = std::chrono::steady_clock::now() - start;
        auto sleep_time = frame_interval - elapsed;
        if (sleep_time > std::chrono::milliseconds(0)) {
            std::this_thread::sleep_for(sleep_time);
        }
    }
    
    std::cout << std::endl;
    std::cout << "shutting down..." << std::endl;
    

    logger->Stop();
    inference->Stop();
    preprocessor->Stop();
    camera->Stop();
    
    std::cout << "processed " << frame_count << " frames" << std::endl;
    
    rclcpp::shutdown();
    return 0;
}
