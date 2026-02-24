#include "YoloOpenCVRT.h"

namespace yolo {
    YoloOpenCVRT::YoloOpenCVRT(const std::string& model_path, bool use_cuda): YoloRuntime("OpenCV::DNN") {
        __net = cv::dnn::readNet(model_path);
        if (use_cuda) {
            __net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            __net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        }
    }
    
    YoloOpenCVRT::~YoloOpenCVRT() {

    }

    std::vector<cv::Mat> YoloOpenCVRT::inference(const cv::Mat& blob) {
        std::vector<cv::Mat> outputs;
        __net.setInput(blob);
        __net.forward(outputs, __net.getUnconnectedOutLayersNames());
        return outputs;
    }
}