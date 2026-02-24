#include <sstream>
#include "YoloConfig.h"
#include "YoloUtils.h"
#include "nlohmann/json.hpp"
using json = nlohmann::json;

namespace yolo {
    std::string to_string(YoloTaskType task) {
        switch (task) {
            case YoloTaskType::CLS:  return "classification";
            case YoloTaskType::DET:  return "detection";
            case YoloTaskType::SEG:  return "segmentation";
            case YoloTaskType::POSE: return "pose";
            case YoloTaskType::OBB:  return "obb";
            default: return "unknown";
        }
    }

    std::string to_string(YoloVersion version) {
        switch (version) {
            case YoloVersion::YOLO5:  return "yolov5";
            case YoloVersion::YOLO5U: return "yolov5u(anchor-free)";
            case YoloVersion::YOLO8:  return "yolov8";
            case YoloVersion::YOLO11: return "yolov11";
            case YoloVersion::YOLO26: return "yolov26";
            default: return "unknown";
        }
    }

    std::string to_string(YoloTargetRT target_rt) {
        switch (target_rt) {
            case YoloTargetRT::OPENCV_CPU:  return "opencv::dnn(cpu)";
            case YoloTargetRT::OPENCV_CUDA: return "opencv::dnn(cuda)";
            case YoloTargetRT::ORT_CPU:  return "onnxruntime(cpu)";
            case YoloTargetRT::ORT_CUDA:  return "onnxruntime(cuda)";
            case YoloTargetRT::OVN_AUTO:  return "openvino(auto)";
            case YoloTargetRT::OVN_CPU:  return "openvino(cpu)";
            case YoloTargetRT::OVN_GPU:  return "openvino(integrated gpu)";
            case YoloTargetRT::TRT:  return "tensorrt";
            case YoloTargetRT::RKNN: return "rknn";
            default: return "unknown";
        }
    }
    
    std::string to_string(const YoloConfig& cfg) {
        std::ostringstream oss;
        oss << "########### YoloConfig ###########" << std::endl;
        oss << "description       : " << cfg.desc                 << std::endl;
        oss << "model path        : " << cfg.model_path           << std::endl;
        oss << "task              : " << to_string(cfg.task)      << std::endl;
        oss << "yolo version      : " << to_string(cfg.version)   << std::endl;
        oss << "yolo runtime      : " << to_string(cfg.target_rt) << std::endl;
        oss << "batch size        : " << cfg.batch_size  << std::endl;
        oss << "input width       : " << cfg.input_w     << std::endl;
        oss << "input height      : " << cfg.input_h     << std::endl;
        oss << "num classes       : " << cfg.num_classes << std::endl;
        oss << "num kpts          : " << cfg.num_kpts    << std::endl;
        oss << "num channels      : " << cfg.num_channels<< std::endl;
        oss << "conf threshold    : " << cfg.conf_thresh << std::endl;
        oss << "iou threshold     : " << cfg.iou_thresh  << std::endl;
        oss << "scale factor      : " << cfg.scale_f     << std::endl;
        oss << "nchw              : " << std::string(cfg.nchw ? "yes" : "no") << std::endl;
        oss << "rgb               : " << std::string(cfg.rgb  ? "yes" : "no") << std::endl;

        auto n = std::min(cfg.names.size(), static_cast<size_t>(5));
        auto cfg_top5_names = std::vector<std::string>(cfg.names.begin(), cfg.names.begin() + n);
        oss << "names(top5)       : " <<  cfg_top5_names;

        return oss.str();
    }
}