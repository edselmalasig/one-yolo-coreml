#include "YoloRuntime.h"


namespace yolo {
    YoloRuntime::YoloRuntime(const std::string& rt_name): __rt_name(rt_name) {

    }
    
    YoloRuntime::~YoloRuntime() {

    }

    std::string YoloRuntime::to_string() {
        return __rt_name;
    }
}