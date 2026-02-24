#include <stdexcept>
#include <numeric> 
#include "YoloClsTask.h"

namespace yolo {
    
    YoloClsTask::YoloClsTask(const YoloConfig& cfg): YoloTask(cfg) {

    }
    
    YoloClsTask::~YoloClsTask() {

    }

    cv::Mat YoloClsTask::softmax(const cv::Mat& logits) {
        assert(logits.dims == 2);

        cv::Mat probs = cv::Mat::zeros(logits.size(), logits.type());

        for (int i = 0; i < logits.rows; ++i) {
            cv::Mat row = logits.row(i);

            double maxVal;
            cv::minMaxLoc(row, nullptr, &maxVal);

            cv::Mat expRow;
            cv::exp(row - maxVal, expRow);

            double sumExp = cv::sum(expRow)[0];
            expRow /= sumExp;

            expRow.copyTo(probs.row(i));
        }
        return probs;
    }

    bool YoloClsTask::is_prob_distribution(const cv::Mat& out, double eps) {
        assert(out.dims == 2);

        for (int i = 0; i < out.rows; ++i) {
            cv::Mat row = out.row(i);

            double minVal, maxVal;
            cv::minMaxLoc(row, &minVal, &maxVal);
            double sum = cv::sum(row)[0];

            if (minVal < 0.0) return false;
            if (maxVal > 1.0) return false;
            if (std::abs(sum - 1.0) > eps) return false;
        }
        return true;
    }

    void YoloClsTask::postprocess_one(
        const std::vector<cv::Mat>& raw_outputs,
        int batch_id,
        cv::Size orig_size,
        LetterBoxInfo lb_info,
        YoloResult& result
    ) {
        std::vector<float> scores;
        std::vector<int> cls_ids;
        if (raw_outputs.empty()) {
            return;
        }

        /* just 1 output head for classification task in Yolo */
        auto& raw_output = raw_outputs[0];
        auto output = is_prob_distribution(raw_output) ? raw_output : softmax(raw_output);

        int num_classes = 0;
        const float* scores_ptr = nullptr;

        if (output.dims == 2) {
            // [batch, num_classes]
            assert(batch_id < output.size[0]);
            num_classes = output.size[1];
            scores_ptr = output.ptr<float>(batch_id);
        }
        else if (output.dims == 1) {
            // [num_classes]
            num_classes = output.size[0];
            scores_ptr = output.ptr<float>();
        }
        else {
            throw std::runtime_error("unsupported output dims for classification task in Yolo!");
            return;
        }

        assert(num_classes == _cfg.num_classes);
        std::vector<int> indices(num_classes);
        std::iota(indices.begin(), indices.end(), 0);

        std::sort(indices.begin(), indices.end(),
                [&](int a, int b) {
                    return scores_ptr[a] > scores_ptr[b];
                });

        scores.reserve(num_classes);
        cls_ids.reserve(num_classes);

        for (int idx : indices) {
            cls_ids.emplace_back(idx);
            scores.emplace_back(scores_ptr[idx]);
        }

        /* fill YoloResult with YoloClsObjs from high score to low */
        for (size_t i = 0; i < num_classes; i++) {
            YoloClsObj cls_obj{cls_ids[i], scores[i], _cfg.names[cls_ids[i]]};
            result.classes.emplace_back(cls_obj);
        }
    } 
}