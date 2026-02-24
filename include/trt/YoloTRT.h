#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include "YoloRuntime.h"

namespace yolo {
    /**
     * @brief
     * Yolo runtime based on tensorrt library.
    */
    class YoloTRT: public YoloRuntime
    {
    private:
        nvinfer1::IRuntime* __runtime = nullptr;
        nvinfer1::ICudaEngine* __engine = nullptr;
        nvinfer1::IExecutionContext* __context = nullptr;

        std::vector<void*> __device_buffers;
        std::vector<std::vector<int64_t>> __output_shapes;

        void allocate_buffers();
    public:
        YoloTRT(const std::string& model_path);
        ~YoloTRT();
        virtual std::vector<cv::Mat> inference(const cv::Mat& blob) override;
    };
}