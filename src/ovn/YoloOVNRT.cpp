#include "ovn/YoloOVNRT.h"

namespace yolo {
    // docs.openvino.ai/2025/openvino-workflow/running-inference.html
    YoloOVNRT::YoloOVNRT(
        const std::string& model_path, 
        const std::string& device):
        YoloRuntime("OpenVINO") {
        auto model = __core.read_model(model_path);

        ov::AnyMap config;
        config[ov::hint::performance_mode.name()] =
            ov::hint::PerformanceMode::LATENCY;

        __compiled_model = __core.compile_model(model, device, config);
    }
    
    YoloOVNRT::~YoloOVNRT() {

    }

    std::vector<cv::Mat> YoloOVNRT::inference(const cv::Mat& blob) {
        // [batch, 3, input_h, input_w] or [batch, input_h, input_w, 3]
        assert(blob.isContinuous());
        assert(blob.type() == CV_32F);
        assert(blob.dims == 4);

        int d0 = blob.size[0];
        int d1 = blob.size[1];
        int d2 = blob.size[2];
        int d3 = blob.size[3];

        auto infer_request =
            __compiled_model.create_infer_request();
        
        // zero copy
        ov::Tensor input_tensor(
            ov::element::f32,
            { (size_t)d0, (size_t)d1,
            (size_t)d2, (size_t)d3 },
            const_cast<float*>(
                reinterpret_cast<const float*>(blob.data)
            )
        );

        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();

        std::vector<cv::Mat> outputs;
        const auto& output_ports = __compiled_model.outputs();

        for (size_t i = 0; i < output_ports.size(); ++i) {
            ov::Tensor out_tensor = infer_request.get_output_tensor(i);
            const ov::Shape& shape = out_tensor.get_shape();

            float* out_data = out_tensor.data<float>();

            int mat_dims = static_cast<int>(shape.size());
            std::vector<int> mat_sizes(mat_dims);
            for (int d = 0; d < mat_dims; ++d)
                mat_sizes[d] = static_cast<int>(shape[d]);

            cv::Mat out_mat(
                mat_dims,
                mat_sizes.data(),
                CV_32F,
                out_data
            );
            // clone to own the buffer data
            outputs.push_back(out_mat.clone());
        }

        return outputs;
    }
}