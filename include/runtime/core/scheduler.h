#pragma once

#include <vector>

#include "runtime/components/component_interface.h"
#include "runtime/core/status.h"

namespace ptk::core {

class RuntimeContext;

//scheduler that drives component tick loops via manual execution
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
};

} //namespace ptk::core