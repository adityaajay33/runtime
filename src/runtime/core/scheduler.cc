#include "runtime/core/scheduler.h"
#include "runtime/core/runtime_context.h"

namespace ptk::core {

Scheduler::Scheduler() 
    : context_(nullptr), 
      components_(), 
      running_(false), 
      tick_(0),
      timer_(nullptr),
      timer_mode_(false) {}

Scheduler::~Scheduler() {
    Stop();
}

Status Scheduler::Init(RuntimeContext* context) {
    if (context == nullptr) {
        return Status(StatusCode::kInvalidArgument, "Context is null");
    }
    if (context_ != nullptr) {
        return Status(StatusCode::kFailedPrecondition, "Scheduler::Init() called more than once");
    }
    context_ = context;
    return Status::Ok();
}

Status Scheduler::AddComponent(components::ComponentInterface* component) {
    if (!context_) {
        return Status(StatusCode::kFailedPrecondition,
                      "Scheduler::Init must be called before AddComponent");
    }
    if (component == nullptr) {
        return Status(StatusCode::kInvalidArgument, "Component is null");
    }
    components_.push_back(component);
    return Status::Ok();
}

Status Scheduler::Start() {
    if (!context_) {
        return Status(StatusCode::kFailedPrecondition,
                      "Scheduler::Init must be called before Start");
    }
    if (running_) {
        return Status(StatusCode::kFailedPrecondition, "Scheduler is already running");
    }
    if (components_.empty()) {
        return Status(StatusCode::kFailedPrecondition, "No components to run");
    }

    for (auto* c : components_) {
        Status s = c->Init(context_);
        if (!s.ok()) {
            return s;
        }
        s = c->Start();
        if (!s.ok()) {
            return s;
        }
    }

    tick_ = 0;
    running_ = true;
    return Status::Ok();
}

void Scheduler::Stop() {
    StopTimerMode();
    
    if (!running_) {
        return;
    }

    for (auto* c : components_) {
        c->Stop();
    }

    running_ = false;
}

void Scheduler::DoTick() {
    if (!running_) {
        return;
    }
    ++tick_;
    for (auto* c : components_) {
        c->Tick();
    }
}

void Scheduler::RunLoop(int num_ticks) {
    if (!running_) {
        return;
    }
    
    if (num_ticks < 0) {
        //run indefinitely until Stop() is called
        while (running_) {
            DoTick();
        }
    } else {
        for (int i = 0; i < num_ticks && running_; ++i) {
            DoTick();
        }
    }
}

Status Scheduler::StartTimerMode(rclcpp::Node* node, int hz) {
    if (!running_) {
        return Status(StatusCode::kFailedPrecondition, 
                      "Scheduler must be started before enabling timer mode");
    }
    if (timer_mode_) {
        return Status(StatusCode::kFailedPrecondition, "Timer mode already active");
    }
    if (node == nullptr) {
        return Status(StatusCode::kInvalidArgument, "Node is null");
    }
    if (hz <= 0) {
        return Status(StatusCode::kInvalidArgument, "Hz must be positive");
    }
    
    auto period = std::chrono::milliseconds(1000 / hz);
    timer_ = node->create_wall_timer(period, [this]() { DoTick(); });
    timer_mode_ = true;
    
    return Status::Ok();
}

void Scheduler::StopTimerMode() {
    if (timer_) {
        timer_->cancel();
        timer_.reset();
    }
    timer_mode_ = false;
}

} //namespace ptk::core