#pragma once

#include <vector>
#include <memory>
#include <chrono>
#include <functional>

#include "runtime/components/component_interface.h"
#include "runtime/core/status.h"
#include <rclcpp/rclcpp.hpp>

namespace ptk::core {

class RuntimeContext;

//scheduler that drives component tick loops
//supports both manual loop and ros timer-based execution
class Scheduler {
public:
    Scheduler();
    ~Scheduler();

    Status Init(RuntimeContext* context);
    Status AddComponent(components::ComponentInterface* component);
    Status Start();
    void Stop();
    
    //manual tick loop - blocks until num_ticks reached or stopped
    void RunLoop(int num_ticks = -1);
    
    //timer-based execution - requires ros node for timer creation
    //hz is the tick frequency (e.g. 30 for 30fps)
    Status StartTimerMode(rclcpp::Node* node, int hz);
    void StopTimerMode();
    
    //check if scheduler is running
    bool IsRunning() const { return running_; }
    
    //get tick count
    int TickCount() const { return tick_; }

private:
    void DoTick();
    
    RuntimeContext* context_;
    std::vector<components::ComponentInterface*> components_;
    bool running_;
    int tick_;
    
    //ros timer for timer-based mode
    rclcpp::TimerBase::SharedPtr timer_;
    bool timer_mode_;
};

} //namespace ptk::core