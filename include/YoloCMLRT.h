#pragma once
#ifdef BUILD_WITH_CML

#include "YoloRuntime.h"

namespace yolo {
    class YoloCMLRT : public YoloRuntime {
    private:
        void* __model         = nullptr;   // MLModel*
        bool  __inputIsImage  = false;     // true if model expects CVPixelBuffer input
    public:
        YoloCMLRT(const std::string& model_path);
        ~YoloCMLRT();
        virtual std::vector<cv::Mat> inference(const cv::Mat& blob) override;
    };
}

#endif // BUILD_WITH_CML