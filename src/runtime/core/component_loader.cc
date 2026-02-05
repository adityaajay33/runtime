#include "runtime/core/component_loader.h"
#include <fstream>
#include <sstream>

namespace ptk::core {

ComponentLoader::ComponentLoader() : factories_() {}

void ComponentLoader::RegisterComponent(const std::string& name, ComponentFactory factory) {
    factories_[name] = std::move(factory);
}

std::shared_ptr<components::ComponentInterface> ComponentLoader::CreateComponent(
    const std::string& name,
    const std::map<std::string, rclcpp::ParameterValue>& params) {
    
    auto it = factories_.find(name);
    if (it == factories_.end()) {
        return nullptr;
    }
    
    //create node options with parameter overrides
    rclcpp::NodeOptions opts;
    for (const auto& [key, value] : params) {
        opts.append_parameter_override(key, value);
    }
    
    return it->second(opts);
}

Status ComponentLoader::LoadFromYaml(
    const std::string& yaml_path,
    std::vector<std::shared_ptr<components::ComponentInterface>>& components) {
    
    //basic yaml parsing - expects format:
    //components:
    //  - name: component_name
    //    params:
    //      param1: value1
    
    std::ifstream file(yaml_path);
    if (!file.is_open()) {
        return Status(StatusCode::kNotFound, "Could not open yaml file: " + yaml_path);
    }
    
    std::string line;
    std::string current_component;
    std::map<std::string, rclcpp::ParameterValue> current_params;
    bool in_params = false;
    
    while (std::getline(file, line)) {
        //skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        //trim leading whitespace
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) {
            continue;
        }
        line = line.substr(start);
        
        //check for component name
        if (line.find("name:") == 0) {
            //save previous component if any
            if (!current_component.empty()) {
                auto comp = CreateComponent(current_component, current_params);
                if (comp) {
                    components.push_back(comp);
                }
            }
            
            current_component = line.substr(5);
            //trim whitespace
            size_t s = current_component.find_first_not_of(" \t");
            size_t e = current_component.find_last_not_of(" \t");
            if (s != std::string::npos) {
                current_component = current_component.substr(s, e - s + 1);
            }
            current_params.clear();
            in_params = false;
        }
        else if (line.find("params:") == 0) {
            in_params = true;
        }
        else if (in_params && line.find(":") != std::string::npos) {
            //parse param: value
            size_t colon = line.find(":");
            std::string key = line.substr(0, colon);
            std::string val = line.substr(colon + 1);
            
            //trim
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            val.erase(0, val.find_first_not_of(" \t"));
            val.erase(val.find_last_not_of(" \t") + 1);
            
            //try to parse as different types
            if (val == "true" || val == "True") {
                current_params[key] = rclcpp::ParameterValue(true);
            } else if (val == "false" || val == "False") {
                current_params[key] = rclcpp::ParameterValue(false);
            } else {
                //try int, then double, then string
                try {
                    int i = std::stoi(val);
                    current_params[key] = rclcpp::ParameterValue(i);
                } catch (...) {
                    try {
                        double d = std::stod(val);
                        current_params[key] = rclcpp::ParameterValue(d);
                    } catch (...) {
                        current_params[key] = rclcpp::ParameterValue(val);
                    }
                }
            }
        }
    }
    
    //save last component
    if (!current_component.empty()) {
        auto comp = CreateComponent(current_component, current_params);
        if (comp) {
            components.push_back(comp);
        }
    }
    
    return Status::Ok();
}

std::vector<std::string> ComponentLoader::GetRegisteredComponents() const {
    std::vector<std::string> names;
    names.reserve(factories_.size());
    for (const auto& [name, _] : factories_) {
        names.push_back(name);
    }
    return names;
}

bool ComponentLoader::HasComponent(const std::string& name) const {
    return factories_.find(name) != factories_.end();
}

//global singleton
ComponentLoader& GetGlobalComponentLoader() {
    static ComponentLoader loader;
    return loader;
}

} //namespace ptk::core
