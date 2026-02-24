#include <stdexcept>
#include "Yolo.h"
#include "YoloClsTask.h"
#include "YoloDetTask.h"
#include "YoloSegTask.h"
#include "YoloPoseTask.h"
#include "YoloObbTask.h"

namespace yolo {
    Yolo::Yolo(const YoloConfig& cfg): __cfg(cfg) {
        if (__cfg.num_classes != __cfg.names.size()) {
            throw std::invalid_argument("num_classes != labels.size() in YoloConfig!");
        }

        switch (__cfg.task) {
        case YoloTaskType::CLS:
            __task = std::make_shared<yolo::YoloClsTask>(cfg);
            break;
        case YoloTaskType::DET:
            __task = std::make_shared<yolo::YoloDetTask>(cfg);
            break;
        case YoloTaskType::SEG:
            __task = std::make_shared<yolo::YoloSegTask>(cfg);
            break;
        case YoloTaskType::POSE:
            __task = std::make_shared<yolo::YoloPoseTask>(cfg);
            break;
        case YoloTaskType::OBB:
            __task = std::make_shared<yolo::YoloObbTask>(cfg);
            break;
        default:
            throw std::invalid_argument("invalid YoloTaskType parameter when initializing Yolo!");
            break;
        }
    }
    
    Yolo::~Yolo() {

    }

    YoloResult Yolo::predict(const cv::Mat& image) {
        return predict(std::vector<cv::Mat>{image})[0];
    }

    YoloResult Yolo::operator()(const cv::Mat& image) {
        return (*this)(std::vector<cv::Mat>{image})[0];
    }

    std::vector<YoloResult> Yolo::predict(const std::vector<cv::Mat>& images) {
        if (__cfg.batch_size && __cfg.batch_size != images.size()) {
            throw std::runtime_error("got invalid batch size when calling Yolo::predict()!");
        }
        
        return (*__task)(images);
    }

    std::vector<YoloResult> Yolo::operator()(const std::vector<cv::Mat>& images) {
        return predict(images);
    }

    std::string Yolo::info(bool print) {
        auto cfg_summary = to_string(__cfg);

        if (print) {
            std::cout << cfg_summary << std::endl;
        }
        return cfg_summary;
    }
}