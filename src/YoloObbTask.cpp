#include "YoloObbTask.h"

namespace yolo {
    
    YoloObbTask::YoloObbTask(const YoloConfig& cfg): YoloTask(cfg)
    {
    }
    
    YoloObbTask::~YoloObbTask()
    {
    }

    void YoloObbTask::collect_rboxes_yolo5u_8_11(
        const cv::Mat&                output,
        int                           batch_id,
        const cv::Size&               orig_size,
        const LetterBoxInfo&          lb,
        std::vector<cv::RotatedRect>& rboxes,
        std::vector<float>&           scores,
        std::vector<int>&             cls_ids
    ) {
        int num_preds = 0;
        int elem_dim  = 0;
        const float* data = nullptr;

        if (output.dims == 3) {
            // [batch, elem_dim, num_preds]
            assert(batch_id < output.size[0]);
            num_preds = output.size[2];
            elem_dim  = output.size[1];
            data = output.ptr<float>(batch_id);
        }
        else if (output.dims == 2) {
            // [elem_dim, num_preds]
            num_preds = output.size[1];
            elem_dim  = output.size[0];
            data = output.ptr<float>();
        }
        else {
            throw std::runtime_error("unsupported output dims for obb task in Yolo!");
            return;
        }

         // [cx, cy, w, h, num_classes, angle]
        const int num_classes = elem_dim - 5;
        assert(num_classes == _cfg.num_classes);
        YoloUtils utils;

        for (int i = 0; i < num_preds; ++i) {
            float cx = data[0 * num_preds + i];
            float cy = data[1 * num_preds + i];
            float w  = data[2 * num_preds + i];
            float h  = data[3 * num_preds + i];
            float angle  = data[(elem_dim - 1) * num_preds + i];  // the last 1 is angle

            int   best_cls = 0;
            float best_score = data[4 * num_preds + i];

            for (int c = 1; c < num_classes; ++c) {
                float s = data[(4 + c)* num_preds + i];
                if (s > best_score) {
                    best_score = s;
                    best_cls = c;
                }
            }

            if (best_score < _cfg.conf_thresh) continue;

            cv::RotatedRect rbox = utils.decode_rbox(
                cx, cy, w, h, angle,
                lb, orig_size
            );

            if (rbox.center.x <= 0 || rbox.center.y <= 0) {
                continue;
            }
            
            rboxes.emplace_back(rbox);
            scores.emplace_back(best_score);
            cls_ids.emplace_back(best_cls);
        }       
    }

    void YoloObbTask::collect_rboxes_yolo26(
        const cv::Mat&                output,
        int                           batch_id,
        const cv::Size&               orig_size,
        const LetterBoxInfo&          lb,
        std::vector<cv::RotatedRect>& rboxes,
        std::vector<float>&           scores,
        std::vector<int>&             cls_ids
    ) {
        int num_preds = 0;
        // const value
        const int elem_dim = 7;
        const float* data = nullptr;

        if (output.dims == 3) {
            // [batch, num_preds, 7]
            // [batch, num_preds, (x1, y1, x2, y2, conf, cls_id, angle)]
            assert(output.size[2] == elem_dim);
            assert(batch_id < output.size[0]);
            num_preds = output.size[1];
            data = output.ptr<float>(batch_id);
        }
        else if (output.dims == 2) {
            // [num_preds, 7]
            // [num_preds, (x1, y1, x2, y2, conf, cls_id, angle)]
            assert(output.size[1] == elem_dim);
            num_preds = output.size[0];
            data = output.ptr<float>();
        }
        else {
            throw std::runtime_error("unsupported output dims for obb task in Yolo!");
            return;
        }

        YoloUtils utils;
        for (int i = 0; i < num_preds; ++i) {
            const float* ptr = data + i * elem_dim;

            float x1   = ptr[0];
            float y1   = ptr[1];
            float x2   = ptr[2];
            float y2   = ptr[3];
            float conf = ptr[4];
            int   cls  = static_cast<int>(ptr[5]);
            float angle= ptr[6];

            if (conf < _cfg.conf_thresh) {
                continue;
            }

            if (cls < 0 || cls >= _cfg.num_classes) {
                continue;
            }

            cv::RotatedRect rbox = utils.decode_rbox(
                x1, y1, x2, y2, angle,
                lb, orig_size
            );

            if (rbox.center.x <= 0 || rbox.center.y <= 0) {
                continue;
            }

            rboxes.emplace_back(rbox);
            scores.emplace_back(conf);
            cls_ids.emplace_back(cls);
        }
    }

    void YoloObbTask::postprocess_one(
        const std::vector<cv::Mat>& raw_outputs,
        int batch_id,
        cv::Size orig_size,
        LetterBoxInfo lb_info,
        YoloResult& result
    ) {
        if (raw_outputs.empty()) {
            return;
        }

        /* just 1 output head for obb task in Yolo */
        const cv::Mat& output = raw_outputs[0];

        /* collect rotated boxes to be used later */
        std::vector<cv::RotatedRect> tmp_rboxes;
        std::vector<float>           tmp_scores;
        std::vector<int>             tmp_cls_ids;
        std::vector<int> keep_indices;
        YoloUtils        utils;
        if (_cfg.version == YoloVersion::YOLO5) {
            // anchor-based
            /*
            collect_boxes_yolo5(
                output, batch_id, orig_size, lb_info,
                tmp_boxes, tmp_scores, tmp_cls_ids
            );
            utils.class_aware_nms(
                tmp_boxes,
                tmp_scores,
                tmp_cls_ids,
                _cfg.conf_thresh,
                _cfg.iou_thresh,
                keep_indices
            ); */
        }
        else if (_cfg.version == YoloVersion::YOLO5U
        || _cfg.version == YoloVersion::YOLO8 
        || _cfg.version == YoloVersion::YOLO11) {
            // anchor-free
            collect_rboxes_yolo5u_8_11(
                output, batch_id, orig_size, lb_info,
                tmp_rboxes, tmp_scores, tmp_cls_ids
            );
            utils.class_aware_nms(
                tmp_rboxes,
                tmp_scores,
                tmp_cls_ids,
                _cfg.conf_thresh,
                _cfg.iou_thresh,
                keep_indices
            );
        }
        else if (_cfg.version == YoloVersion::YOLO26) {
            // anchor-free & nms-free
            collect_rboxes_yolo26(
                output, batch_id, orig_size, lb_info,
                tmp_rboxes, tmp_scores, tmp_cls_ids
            );
            keep_indices.resize(tmp_rboxes.size());
            std::iota(keep_indices.begin(), keep_indices.end(), 0);  // keep all
        }
        else {
            /* not supported */
            return;
        }

        /* final rotated boxes */
        std::vector<cv::RotatedRect> rboxes;
        std::vector<float> scores;
        std::vector<int> cls_ids;

        for (int idx : keep_indices) {
            rboxes.emplace_back(tmp_rboxes[idx]);
            scores.emplace_back(tmp_scores[idx]);
            cls_ids.emplace_back(tmp_cls_ids[idx]);
        }

        /* fill YoloResult with YoloObbObjs */
        for (size_t i = 0; i < keep_indices.size(); i++) {
            YoloObbObj ob_obj {rboxes[i],  cls_ids[i], scores[i], _cfg.names[cls_ids[i]]};
            result.obbs.emplace_back(ob_obj);
        }
    }     
}