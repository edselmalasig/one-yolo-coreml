#include <fstream>
#include "trt/YoloTRT.h"

namespace yolo {
    class TRTLogger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING)
                std::cout << "[TensorRT] " << msg << std::endl;
        }
    } gLogger;

    YoloTRT::YoloTRT(const std::string& model_path):
        YoloRuntime("TensorRT") {
        std::ifstream file(model_path, std::ios::binary);

        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);

        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);
        file.close();

        // create runtime / engine / context
        __runtime = nvinfer1::createInferRuntime(gLogger);
        __engine = __runtime->deserializeCudaEngine(engine_data.data(), size);
        __context = __engine->createExecutionContext();

        allocate_buffers();
    }
    
    YoloTRT::~YoloTRT() {
        for (void* buf : __device_buffers)
            cudaFree(buf);

        if (__context) __context->destroy();
        if (__engine) __engine->destroy();
        if (__runtime) __runtime->destroy();
    }

    void YoloTRT::allocate_buffers() {
        int nbBindings = __engine->getNbBindings();
        __device_buffers.resize(nbBindings);

        for (int i = 0; i < nbBindings; ++i) {
            auto dims = __engine->getBindingDimensions(i);
            size_t vol = 1;

            for (int d = 0; d < dims.nbDims; ++d)
                vol *= dims.d[d];

            size_t bytes = vol * sizeof(float);

            cudaMalloc(&__device_buffers[i], bytes);

            if (!__engine->bindingIsInput(i)) {
                std::vector<int64_t> shape;
                for (int d = 0; d < dims.nbDims; ++d)
                    shape.push_back(dims.d[d]);

                __output_shapes.push_back(shape);
            }
        }
    }

    std::vector<cv::Mat> YoloTRT::inference(const cv::Mat& blob) {
        // [batch, 3, input_h, input_w] or [batch, input_h, input_w, 3]
        assert(blob.isContinuous());
        assert(blob.type() == CV_32F);
        assert(blob.dims == 4);

        int d0 = blob.size[0];
        int d1 = blob.size[1];
        int d2 = blob.size[2];
        int d3 = blob.size[3];

        size_t input_bytes =
            d0 * d1 * d2 * d3 * sizeof(float);
        cudaMemcpy(
            __device_buffers[0],
            blob.data,
            input_bytes,
            cudaMemcpyHostToDevice
        );

        __context->enqueueV2(
            __device_buffers.data(),
            0,      // stream
            nullptr
        );

        std::vector<cv::Mat> outputs;
        int output_index = 0;
        for (int i = 0; i < __engine->getNbBindings(); ++i) {
            if (__engine->bindingIsInput(i))
                continue;

            auto dims = __engine->getBindingDimensions(i);

            size_t vol = 1;
            for (int d = 0; d < dims.nbDims; ++d)
                vol *= dims.d[d];

            std::vector<float> host_buffer(vol);

            cudaMemcpy(
                host_buffer.data(),
                __device_buffers[i],
                vol * sizeof(float),
                cudaMemcpyDeviceToHost
            );

            // TensorRT dims → cv::Mat
            std::vector<int> mat_sizes;
            for (int d = 0; d < dims.nbDims; ++d)
                mat_sizes.push_back(dims.d[d]);

            cv::Mat out_mat(
                dims.nbDims,
                mat_sizes.data(),
                CV_32F,
                host_buffer.data()
            );
            // clone to own the buffer data
            outputs.push_back(out_mat.clone());
            output_index++;
        }

        return outputs;
    }
}