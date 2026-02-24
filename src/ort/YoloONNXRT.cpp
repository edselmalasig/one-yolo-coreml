#include "ort/YoloONNXRT.h"

namespace yolo {
    YoloONNXRT::YoloONNXRT(
        const std::string& model_path, 
        bool use_cuda):
        YoloRuntime("ONNXRuntime") {
        // options
        __session_options.SetIntraOpNumThreads(1);
        __session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        if (use_cuda) {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;

            __session_options.AppendExecutionProvider_CUDA(cuda_options);
        }
        
        __session = Ort::Session(__env, model_path.c_str(), __session_options);
    }
    
    YoloONNXRT::~YoloONNXRT() {

    }

    std::vector<cv::Mat> YoloONNXRT::inference(const cv::Mat& blob) {
        std::vector<cv::Mat> outputs;

        // refer to cv::dnn::Net::forward
        ort_forward(blob, outputs);
        return outputs;
    }

    void YoloONNXRT::ort_forward(
        const cv::Mat& input_4d,
        std::vector<cv::Mat>& outputs
    ) {
        // [batch, 3, input_h, input_w] or [batch, input_h, input_w, 3]
        assert(input_4d.isContinuous());
        assert(input_4d.type() == CV_32F);
        assert(input_4d.dims == 4);

        // get input shape
        std::vector<int64_t> input_shape(4);
        for (int i = 0; i < 4; ++i) {
            input_shape[i] = input_4d.size[i];
        }
        auto input_tensor_size = input_4d.total();

        // cv::Mat → ORT Tensor (zero copy)
        Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator,
                OrtMemTypeDefault);
        Ort::Value input_tensor =
            Ort::Value::CreateTensor<float>(
                memory_info,
                (float*)input_4d.data,
                input_tensor_size,
                input_shape.data(),
                input_shape.size());

        //  get input / output num & names
        auto input_names_str = __session.GetInputNames();
        auto output_names_str = __session.GetOutputNames();
        auto num_inputs  = input_names_str.size();   // 1 for Yolo
        auto num_outputs = output_names_str.size();
        std::vector<const char*> input_names;
        std::vector<const char*> output_names;

        for (auto& s : input_names_str)
            input_names.push_back(s.c_str());
        for (auto& s : output_names_str)
            output_names.push_back(s.c_str());

        // run with input tensor and get output tensors
        auto output_tensors = __session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            &input_tensor,
            1,
            output_names.data(),
            num_outputs);

        // ORT Tensor → cv::Mat (should copy buffer data)
        outputs.clear();
        outputs.reserve(num_outputs);
        for (size_t i = 0; i < num_outputs; ++i) {
            auto& out_tensor = output_tensors[i];

            auto shape_info = out_tensor.GetTensorTypeAndShapeInfo();
            auto shape = shape_info.GetShape();
            int dims = shape.size();

            std::vector<int> cv_sizes(dims);
            for (int d = 0; d < dims; ++d)
                cv_sizes[d] = static_cast<int>(shape[d]);

            float* data_ptr = out_tensor.GetTensorMutableData<float>();
            cv::Mat out_mat(
                dims,
                cv_sizes.data(),
                CV_32F,
                data_ptr);
            
            // clone to own the buffer data!
            outputs.emplace_back(out_mat.clone());
        }
    }
}