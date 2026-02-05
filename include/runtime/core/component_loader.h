#pragma once

#include <string>
#include <memory>
#include <map>
#include <functional>
#include <vector>

#include "runtime/core/status.h"
#include "runtime/components/component_interface.h"
#include <rclcpp/rclcpp.hpp>

namespace ptk::core {

//utility for loading and configuring components
class ComponentLoader {
public:
    using ComponentFactory = std::function<std::shared_ptr<components::ComponentInterface>(
        const rclcpp::NodeOptions&)>;
    
    ComponentLoader();
    ~ComponentLoader() = default;
    
    //register a component factory by name
    void RegisterComponent(const std::string& name, ComponentFactory factory);
    
    //create a component by name with optional parameter overrides
    std::shared_ptr<components::ComponentInterface> CreateComponent(
        const std::string& name,
        const std::map<std::string, rclcpp::ParameterValue>& params = {});
    
    //load pipeline configuration from yaml file
    //returns list of created components in order
    Status LoadFromYaml(const std::string& yaml_path,
                        std::vector<std::shared_ptr<components::ComponentInterface>>& components);
    
    //get list of registered component names
    std::vector<std::string> GetRegisteredComponents() const;
    
    //check if a component is registered
    bool HasComponent(const std::string& name) const;

private:
    std::map<std::string, ComponentFactory> factories_;
};

//singleton instance for global registration
ComponentLoader& GetGlobalComponentLoader();

//helper macro to register components
#define PTK_REGISTER_COMPONENT(loader, name, type) \
    loader.RegisterComponent(name, [](const rclcpp::NodeOptions& opts) { \
        return std::make_shared<type>(opts); \
    })

} //namespace ptk::core
