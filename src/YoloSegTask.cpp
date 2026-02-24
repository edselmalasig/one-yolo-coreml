#include "YoloSegTask.h"

namespace yolo {
    YoloSegTask::YoloSegTask(const YoloConfig& cfg): YoloTask(cfg)
    {
    }
    
    YoloSegTask::~YoloSegTask()
    {
    }

    cv::Mat YoloSegTask::sigmoid(const cv::Mat& x) {
        cv::Mat result;
        cv::exp(-x, result);
        result = 1.0f / (1.0f + result);
        return result;
    }

    cv::Mat YoloSegTask::process_mask_one(
        const cv::Mat&   protos,      // [batch, _cfg.num_channels, _cfg.input_h/4, _cfg.input_w/4]
        int              batch_id,
        const cv::Mat&   coeff,       // [1, _cfg.num_channels]
        const cv::Rect&  bbox,           
        const cv::Size&  input_size,
        const cv::Point& padding,
        const float x_scale,
        const float y_scale
    ) {
        // get proto by batch id
        auto un_const_protos = const_cast<cv::Mat&>(protos);
        auto proto_ptr = un_const_protos.ptr<float>(batch_id); 
        int proto_s[] = {protos.size[1], protos.size[2], protos.size[3]};
        // [_cfg.num_channels, _cfg.input_h/4, _cfg.input_w/4]
        cv::Mat proto(3, proto_s, CV_32F, proto_ptr);

        auto h_proto = proto.size[1];
        auto w_proto = proto.size[2];

        // [_cfg.num_channels, _cfg.input_h/4 * _cfg.input_w/4]
        cv::Mat proto_flat = proto.reshape(1, _cfg.num_channels);

        // [1,_cfg.num_channels] × [_cfg.num_channels, _cfg.input_h/4 * _cfg.input_w/4] → [1, _cfg.input_h/4 * _cfg.input_w/4]
        cv::Mat mask_raw = coeff * proto_flat;

        // [_cfg.input_h/4, _cfg.input_w/4]
        cv::Mat mask_2d = mask_raw.reshape(0, h_proto);

        // 6. Sigmoid
        cv::Mat mask_sigmoid = sigmoid(mask_2d);

        // 7. 原图 bbox → 输入图坐标（含 padding）
        float x1_in = bbox.x * x_scale + padding.x;
        float y1_in = bbox.y * y_scale + padding.y;
        float x2_in = (bbox.x + bbox.width) * x_scale + padding.x;
        float y2_in = (bbox.y + bbox.height) * y_scale + padding.y;

        // 8. 输入图 → proto 坐标（/4）
        float x1_p = x1_in / 4.0f;
        float y1_p = y1_in / 4.0f;
        float x2_p = x2_in / 4.0f;
        float y2_p = y2_in / 4.0f;

        int px1 = std::max(0, (int)floor(x1_p));
        int py1 = std::max(0, (int)floor(y1_p));
        int px2 = std::min(w_proto, (int)ceil(x2_p));
        int py2 = std::min(h_proto, (int)ceil(y2_p));

        if (px2 <= px1 || py2 <= py1) {
            return cv::Mat::zeros(bbox.size(), CV_8UC1);
        }

        // 9. 裁剪
        cv::Mat cropped = mask_sigmoid(cv::Rect(px1, py1, px2 - px1, py2 - py1)).clone();

        // 10. 上采样到 bbox 尺寸
        cv::Mat resized;
        resize(cropped, resized, bbox.size(), 0, 0, cv::INTER_LINEAR);

        // 11. 二值化
        cv::Mat mask_bin;
        threshold(resized, mask_bin, 0.5, 255, cv::THRESH_BINARY);
        mask_bin.convertTo(mask_bin, CV_8UC1);

        return mask_bin;
    }

    void YoloSegTask::collect_boxes_yolo5(
        const cv::Mat&         output0,
        const cv::Mat&         output1,
        int                    batch_id,
        const cv::Size&        orig_size,
        const LetterBoxInfo&   lb,
        std::vector<cv::Rect>& boxes,
        std::vector<cv::Mat>&  masks,
        std::vector<float>&    scores,
        std::vector<int>&      cls_ids
    ) {
        // output0: [batch, num_preds, 4 + obj_conf + num_classes + _cfg.num_channels]
        // output1: [batch, _cfg.num_channels, _cfg.input_h/4, _cfg.input_w/4]
        int num_preds = 0;
        int elem_dim  = 0;
        const float* data = nullptr;

        if (output0.dims == 3 && output1.dims == 4) {
            assert(batch_id < output0.size[0]);
            assert(batch_id < output1.size[0]);
            num_preds = output0.size[1];
            elem_dim  = output0.size[2];
            data = output0.ptr<float>(batch_id);
        }
        else {
            throw std::runtime_error("unsupported output dims for segmentation task in Yolo!");
            return;
        }

         // [cx, cy, w, h, obj_conf, num_classes, _cfg.num_channels]
        const int num_classes = elem_dim - 5 - _cfg.num_channels;
        assert(num_classes == _cfg.num_classes);
        YoloUtils utils;

        for (int i = 0; i < num_preds; ++i) {
            const float* ptr = data + i * elem_dim;
            float cx       = ptr[0];
            float cy       = ptr[1];
            float w        = ptr[2];
            float h        = ptr[3];
            float obj_conf = ptr[4];

            if (obj_conf < _cfg.conf_thresh) 
                continue;

            int   best_cls = 0;
            float best_cls_score = ptr[5];
            for (int c = 1; c < num_classes; ++c) {
                if (ptr[5 + c] > best_cls_score) {
                    best_cls_score = ptr[5 + c];
                    best_cls = c;
                }
            }

            float conf = obj_conf * best_cls_score;
            if (conf < _cfg.conf_thresh) 
                continue;

            cv::Rect box = utils.decode_box(
                cx, cy, w, h,
                lb, orig_size
            );

            boxes.emplace_back(box);
            scores.emplace_back(conf);
            cls_ids.emplace_back(best_cls);

            /* here just collect coeff first, extract the final mask after nms for high performance */
            cv::Mat coeff(1, _cfg.num_channels, CV_32F);
            for (int m = 0; m < _cfg.num_channels; ++m) {
                coeff.at<float>(m) = ptr[5 + num_classes + m];
            }

            masks.emplace_back(coeff);
        }  
    }

    void YoloSegTask::collect_boxes_yolo5u_8_11(
        const cv::Mat&         output0,
        const cv::Mat&         output1,
        int                    batch_id,
        const cv::Size&        orig_size,
        const LetterBoxInfo&   lb,
        std::vector<cv::Rect>& boxes,
        std::vector<cv::Mat>&  masks,
        std::vector<float>&    scores,
        std::vector<int>&      cls_ids
    ) {
        // output0: [batch, 4 + num_classes + _cfg.num_channels, num_preds]
        // output1: [batch, _cfg.num_channels, _cfg.input_h/4, _cfg.input_w/4]
        int num_preds = 0;
        int elem_dim  = 0;
        const float* data = nullptr;

        if (output0.dims == 3 && output1.dims == 4) {
            assert(batch_id < output0.size[0]);
            assert(batch_id < output1.size[0]);
            num_preds = output0.size[2];
            elem_dim  = output0.size[1];
            data = output0.ptr<float>(batch_id);
        }
        else {
            throw std::runtime_error("unsupported output dims for segmentation task in Yolo!");
            return;
        }

         // [cx, cy, w, h, num_classes, _cfg.num_channels]
        const int num_classes = elem_dim - 4 - _cfg.num_channels;
        assert(num_classes == _cfg.num_classes);
        YoloUtils utils;

        for (int i = 0; i < num_preds; ++i) {
            float cx = data[0 * num_preds + i];
            float cy = data[1 * num_preds + i];
            float w  = data[2 * num_preds + i];
            float h  = data[3 * num_preds + i];

            int   best_cls = 0;
            float best_score = data[4 * num_preds + i];

            for (int c = 1; c < num_classes; ++c) {
                float s = data[(4 + c) * num_preds + i];
                if (s > best_score) {
                    best_score = s;
                    best_cls = c;
                }
            }

            if (best_score < _cfg.conf_thresh) continue;

            cv::Rect box = utils.decode_box(
                cx, cy, w, h,
                lb, orig_size
            );

            boxes.emplace_back(box);
            scores.emplace_back(best_score);
            cls_ids.emplace_back(best_cls);

            /* here just collect coeff first, extract the final mask after nms for high performance */
            cv::Mat coeff(1, _cfg.num_channels, CV_32F);
            for (int m = 0; m < _cfg.num_channels; ++m) {
                coeff.at<float>(m) = data[(4 + num_classes + m) * num_preds + i];
            }

            masks.emplace_back(coeff);
        }       
    }

    void YoloSegTask::collect_boxes_yolo26(
        const cv::Mat&         output0,
        const cv::Mat&         output1,
        int                    batch_id,
        const cv::Size&        orig_size,
        const LetterBoxInfo&   lb,
        std::vector<cv::Rect>& boxes,
        std::vector<cv::Mat>&  masks,
        std::vector<float>&    scores,
        std::vector<int>&      cls_ids
    ) {
        // output0: [batch, num_preds, 6 + _cfg.num_channels]
        // output1: [batch, _cfg.num_channels, _cfg.input_h/4, _cfg.input_w/4]
        int num_preds = 0;
        // const value
        const int elem_dim = 6 + _cfg.num_channels;
        const float* data = nullptr;

        if (output0.dims == 3 && output1.dims == 4) {
            // [batch, num_preds, 6 + _cfg.num_channels]
            // [batch, num_preds, (x1, y1, x2, y2, conf, cls_id) + _cfg.num_channels]
            assert(batch_id < output0.size[0]);
            assert(batch_id < output1.size[0]);
            num_preds = output0.size[1];
            data = output0.ptr<float>(batch_id);
        }
        else {
            throw std::runtime_error("unsupported output dims for segmentation task in Yolo!");
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

            if (conf < _cfg.conf_thresh) {
                continue;
            }

            if (cls < 0 || cls >= _cfg.num_classes) {
                continue;
            }

            cv::Rect box = utils.decode_box(
                (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1,
                lb, orig_size
            );

            if (box.width <= 0 || box.height <= 0) {
                continue;
            }


            boxes.emplace_back(box);
            scores.emplace_back(conf);
            cls_ids.emplace_back(cls);

            /* here just collect coeff first, extract the final mask after nms(keep the same action for nms-free) 
               for high performance */
            cv::Mat coeff(1, _cfg.num_channels, CV_32F);
            for (int m = 0; m < _cfg.num_channels; ++m) {
                coeff.at<float>(m) = ptr[6 + m];
            }

            masks.emplace_back(coeff);
        }        
    }

    void YoloSegTask::postprocess_one(
        const std::vector<cv::Mat>& raw_outputs,
        int batch_id,
        cv::Size orig_size,
        LetterBoxInfo lb_info,
        YoloResult& result
    ) {
        if (raw_outputs.empty()) {
            return;
        }

        /* 2 output heads for segmentation task in Yolo (take care of the sequence of mats in raw_outputs) */
        const cv::Mat& output0 = raw_outputs[0].dims == 3 ? raw_outputs[0] : raw_outputs[1];  // output0: depends on YoloVersion
        const cv::Mat& output1 = raw_outputs[1].dims == 4 ? raw_outputs[1] : raw_outputs[0];  // output1: [batch, _cfg.num_channels, _cfg.input_h/4, _cfg.input_w/4]

        /* collect boxes & masks to be used later */
        std::vector<cv::Rect> tmp_boxes;
        std::vector<cv::Mat>  tmp_masks;
        std::vector<float>    tmp_scores;
        std::vector<int>      tmp_cls_ids;
        std::vector<int> keep_indices;
        YoloUtils        utils;
        if (_cfg.version == YoloVersion::YOLO5) {
            // anchor-based
            collect_boxes_yolo5(
                output0, output1, batch_id, orig_size, lb_info,
                tmp_boxes, tmp_masks, tmp_scores, tmp_cls_ids
            );
            utils.class_aware_nms(
                tmp_boxes,
                tmp_scores,
                tmp_cls_ids,
                _cfg.conf_thresh,
                _cfg.iou_thresh,
                keep_indices
            );
        }
        else if (_cfg.version == YoloVersion::YOLO5U
        || _cfg.version == YoloVersion::YOLO8 
        || _cfg.version == YoloVersion::YOLO11) {
            // anchor-free
            collect_boxes_yolo5u_8_11(
                output0, output1, batch_id, orig_size, lb_info,
                tmp_boxes, tmp_masks, tmp_scores, tmp_cls_ids
            );
            utils.class_aware_nms(
                tmp_boxes,
                tmp_scores,
                tmp_cls_ids,
                _cfg.conf_thresh,
                _cfg.iou_thresh,
                keep_indices
            );
        }
        else if (_cfg.version == YoloVersion::YOLO26) {
            // anchor-free & nms-free
            collect_boxes_yolo26(
                output0, output1, batch_id, orig_size, lb_info,
                tmp_boxes, tmp_masks, tmp_scores, tmp_cls_ids
            );
            keep_indices.resize(tmp_boxes.size());
            std::iota(keep_indices.begin(), keep_indices.end(), 0);  // keep all
        }
        else {
            /* not supported */
            return;
        }

        /* final boxes & masks */
        std::vector<cv::Rect> boxes;
        std::vector<cv::Mat> masks;
        std::vector<float> scores;
        std::vector<int> cls_ids;
        for (int idx : keep_indices) {
            boxes.emplace_back(tmp_boxes[idx]);
            scores.emplace_back(tmp_scores[idx]);
            cls_ids.emplace_back(tmp_cls_ids[idx]);

            // extract the final mask using coeff & protos
            auto& coeff = tmp_masks[idx];
            auto& protos = output1;
            auto local_mask = 
                process_mask_one(
                    protos, batch_id, coeff, tmp_boxes[idx],
                    cv::Size(_cfg.input_w, _cfg.input_h),
                    cv::Point(lb_info.pad_w, lb_info.pad_h), lb_info.scale, lb_info.scale);
            // binary mask(0 background / 255 object), has the same size as bbox
            masks.emplace_back(local_mask);
        }

        /* fill YoloResult with YoloSegObjs */
        for (size_t i = 0; i < keep_indices.size(); i++) {
            // calculate & only return the largest contour according to the local mask
            auto& mask = masks[i];
            std::vector<std::vector<cv::Point>> contours;
            cv::Mat hierarchy;
            std::vector<cv::Point> approx_contour;
            cv::findContours(mask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

            if (!contours.empty()) {
                auto maxContourIt = std::max_element(
                    contours.begin(), 
                    contours.end(),
                    [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                        return cv::contourArea(a) < cv::contourArea(b);
                    }
                );
                
                auto largestContour = *maxContourIt;

                // reduce the points to make contour more simple
                double epsilon = _cfg.approx_f * cv::arcLength(largestContour, true);
                cv::approxPolyDP(largestContour, approx_contour, epsilon, true);

                // add offset to match original size (because it's from local mask)
                for (auto& point : approx_contour) {
                    point.x += boxes[i].x;
                    point.y += boxes[i].y;
                }
            }
            
            /* note:
               the mask is local mask(binary value, 0 or 255) and has the same size as bbox of detected object,
               it maybe contains multiple un-connected area.
               we can use contour directly in some situations such as drawing.
            */
            YoloSegObj seg_obj {boxes[i],  cls_ids[i], scores[i], _cfg.names[cls_ids[i]], mask, approx_contour};
            result.segmentations.emplace_back(seg_obj);
        }
    }
}