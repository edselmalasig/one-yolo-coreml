#include <sstream>
#include "YoloResult.h"
#include "YoloUtils.h"


namespace yolo {
    cv::Mat YoloResult::plot(const DrawParam& param) {
        switch (task) {
        case YoloTaskType::CLS: {
            auto top5_        = top5();
            auto top5_confs_  = top5_confs();
            auto top5_labels_ = top5_labels();

            return draw_results(
                orig_image,
                param,
                top5_,
                top5_confs_,
                top5_labels_,
                std::vector<int>(),
                std::vector<float>(),
                std::vector<std::string>(),
                std::vector<cv::Rect>(),
                std::vector<cv::RotatedRect>(),
                std::vector<cv::Mat>(),
                std::vector<std::vector<cv::Point>>(),
                std::vector<std::vector<YoloKeyPoint>>(),
                std::vector<int>(),
                std::vector<std::vector<cv::Point>>());
        }
        case YoloTaskType::DET: {
            auto boxes_   = boxes();
            auto cls_ids_ = cls_ids();
            auto confs_   = confs();
            auto labels_  = labels();
            auto track_ids_    = track_ids();
            auto track_points_ = track_points();

            return draw_results(
                orig_image,
                param,
                std::vector<int>(),
                std::vector<float>(),
                std::vector<std::string>(),
                cls_ids_,
                confs_,
                labels_,
                boxes_,
                std::vector<cv::RotatedRect>(),
                std::vector<cv::Mat>(),
                std::vector<std::vector<cv::Point>>(),
                std::vector<std::vector<YoloKeyPoint>>(),
                track_ids_,
                track_points_);
        }
        case YoloTaskType::SEG: {
            auto boxes_   = boxes();
            auto cls_ids_ = cls_ids();
            auto confs_   = confs();
            auto labels_  = labels();
            auto masks_   = masks();
            auto contours_     = contours();
            auto track_ids_    = track_ids();
            auto track_points_ = track_points();

            return draw_results(
                orig_image,
                param,
                std::vector<int>(),
                std::vector<float>(),
                std::vector<std::string>(),
                cls_ids_,
                confs_,
                labels_,
                boxes_,
                std::vector<cv::RotatedRect>(),
                masks_,
                contours_,
                std::vector<std::vector<YoloKeyPoint>>(),
                track_ids_,
                track_points_);
        }
        case YoloTaskType::POSE: {
            auto boxes_   = boxes();
            auto cls_ids_ = cls_ids();
            auto confs_   = confs();
            auto labels_  = labels();
            auto kpts_    = kpts();
            auto track_ids_    = track_ids();
            auto track_points_ = track_points();

            return draw_results(
                orig_image,
                param,
                std::vector<int>(),
                std::vector<float>(),
                std::vector<std::string>(),
                cls_ids_,
                confs_,
                labels_,
                boxes_,
                std::vector<cv::RotatedRect>(),
                std::vector<cv::Mat>(),
                std::vector<std::vector<cv::Point>>(),
                kpts_,
                track_ids_,
                track_points_);
        }
        case YoloTaskType::OBB: {
            auto rboxes_   = rboxes();
            auto cls_ids_ = cls_ids();
            auto confs_   = confs();
            auto labels_  = labels();

            return draw_results(
                orig_image,
                param,
                std::vector<int>(),
                std::vector<float>(),
                std::vector<std::string>(),
                cls_ids_,
                confs_,
                labels_,
                std::vector<cv::Rect>(),
                rboxes_,
                std::vector<cv::Mat>(),
                std::vector<std::vector<cv::Point>>(),
                std::vector<std::vector<YoloKeyPoint>>(),
                std::vector<int>(),
                std::vector<std::vector<cv::Point>>());
        }
        default:
            throw std::runtime_error("invalid task type in YoloResult!");
            break;
        }
    }

    std::string YoloResult::save(const DrawParam& param) {
        auto plot_img  = plot(param);
        // save
    }

    int YoloResult::show(
        bool block, 
        float scale_f,
        const DrawParam& param, 
        bool show_orig_img, 
        bool show_input_img) {
        auto plot_img  = plot(param);
        scale_f = std::max(0.0f, std::min(1.0f, scale_f));
        if (std::abs(scale_f - 1.0f) > FLT_EPSILON) {
            cv::resize(plot_img, plot_img, cv::Size(), scale_f, scale_f);
        }
        
        cv::imshow("plot-img-" + std::to_string(id) + "(" + std::to_string(plot_img.cols) + "*" + std::to_string(plot_img.rows) + ")", plot_img);
        if (show_orig_img) {
            cv::imshow("orig-img-" + std::to_string(id) + "(" + std::to_string(orig_image.cols) + "*" + std::to_string(orig_image.rows) + ")", orig_image);
        }
        if (show_input_img) {
            cv::imshow("input-img-" + std::to_string(id) + "(" + std::to_string(input_image.cols) + "*" + std::to_string(input_image.rows) + ")", input_image);
        }
        
        auto delay = block ? 0 : 1;
        return cv::waitKey(delay);
    }

    std::string YoloResult::to_csv(bool print) {
        std::ostringstream oss;
        switch (task) {
        case YoloTaskType::CLS: {
            oss << "rank,cls_id,conf,label" << std::endl;
            for (size_t i = 0; i < classes.size(); i++) {
                auto& obj = classes[i];
                oss << (i+1) << "," << obj.cls_id << "," << obj.conf << "," << obj.label;
                if (i + 1 != classes.size()) {
                    oss << std::endl;
                }
            }
            break;
        }
        case YoloTaskType::DET: {
            oss << "id,cls_id,conf,label,track_id" << std::endl;
            for (size_t i = 0; i < detections.size(); i++) {
                auto& obj = detections[i];
                oss << (i+1) << "," << obj.cls_id << "," << obj.conf << "," << obj.label << "," << obj.track_id;
                if (i + 1 != detections.size()) {
                    oss << std::endl;
                }
            }
            break;
        }
        case YoloTaskType::SEG: {
            oss << "id,cls_id,conf,label,track_id" << std::endl;
            for (size_t i = 0; i < segmentations.size(); i++) {
                auto& obj = segmentations[i];
                oss << (i+1) << "," << obj.cls_id << "," << obj.conf << "," << obj.label << "," << obj.track_id;
                if (i + 1 != segmentations.size()) {
                    oss << std::endl;
                }
            }
            break;
        }
        case YoloTaskType::POSE: {
            oss << "id,cls_id,conf,label,track_id" << std::endl;
            for (size_t i = 0; i < poses.size(); i++) {
                auto& obj = poses[i];
                oss << (i+1) << "," << obj.cls_id << "," << obj.conf << "," << obj.label << "," << obj.track_id;
                if (i + 1 != poses.size()) {
                    oss << std::endl;
                }
            }
            break;
        }
        case YoloTaskType::OBB: {
            oss << "id,cls_id,conf,label" << std::endl;
            for (size_t i = 0; i < obbs.size(); i++) {
                auto& obj = obbs[i];
                oss << (i+1) << "," << obj.cls_id << "," << obj.conf << "," << obj.label;
                if (i + 1 != obbs.size()) {
                    oss << std::endl;
                }
            }
            break;
        }
        default:
            throw std::runtime_error("invalid task type in YoloResult!");
            break;
        }
        auto c_str = oss.str();
        if (print) {
            std::cout << c_str << std::endl;
        }
        
        return c_str;
    }

    std::string YoloResult::to_json(bool print, bool indent) {
        std::string j_str = "";
        auto indent_num = indent ? 4 : -1;
        switch (task) {
        case YoloTaskType::CLS: {
            json j = classes;
            j_str  = j.dump(indent_num);
            break;
        }
        case YoloTaskType::DET: {
            json j = detections;
            j_str  = j.dump(indent_num);
            break;
        }
        case YoloTaskType::SEG: {
            json j = segmentations;
            j_str  = j.dump(indent_num);
            break;
        }
        case YoloTaskType::POSE: {
            json j = poses;
            j_str  = j.dump(indent_num);
            break;
        }
        case YoloTaskType::OBB: {
            json j = obbs;
            j_str  = j.dump(indent_num);
            break;
        }
        default:
            throw std::runtime_error("invalid task type in YoloResult!");
            break;
        }
        if (print) {
            std::cout << j_str << std::endl;
        }
        
        return j_str;
    }

    std::string YoloResult::info(bool print) {
        std::ostringstream oss;
        oss << "########### YoloResult ###########" << std::endl;
        oss << "id                : " << id         << std::endl;
        oss << "task              : " << to_string(task)       << std::endl;
        oss << "yolo version      : " << to_string(version)    << std::endl;
        oss << "yolo runtime      : " << to_string(target_rt)  << std::endl;
        oss << "batch size        : " << batch_size << std::endl;
        oss << "input width       : " << input_w    << std::endl;
        oss << "input height      : " << input_h    << std::endl;
        oss << "original size     : " << orig_size.width      << " * " << orig_size.height << std::endl;
        oss << "letterbox         : " << "scale: " << letterbox_info.scale << ", pad_w: " << letterbox_info.pad_w << ", pad_h: " << letterbox_info.pad_h << std::endl;
        oss << "speed             : " << "pre: " << speed[0]  << "ms, infer: " << speed[1] << "ms, post: " << speed[2] << "ms" << std::endl;
        auto n = std::min(names.size(), static_cast<size_t>(5));
        auto top5_names = std::vector<std::string>(names.begin(), names.begin() + n);
        oss << "names(top5)       : " <<  top5_names << std::endl;
        if (task == YoloTaskType::CLS) {
            // labels&confs of top5: label0(conf0), label1(conf1), ...
            auto t5_labels = top5_labels();
            auto t5_confs  = top5_confs();
            auto t5_out = t5_labels[0] + "(" + std::to_string(t5_confs[0]) + ")";
            for (size_t i = 1; i < t5_labels.size(); i++) {
                t5_out += ", " + t5_labels[i] + "(" + std::to_string(t5_confs[i]) + ")";
            }
            oss << "top5              : " << t5_out;
        }
        else if (task == YoloTaskType::OBB) {
            // number of rboxes, which stand for the number of predicted objects.
            oss << "objects count     : " << rboxes().size();
        }
        else {
            // number of boxes, which stand for the number of predicted objects.
            oss << "objects count     : " << boxes().size();
        }
        
        auto summary = oss.str();
        if (print) {
            std::cout << summary << std::endl;
        }
        return summary;
    }

    int YoloResult::top1() const {
        if (task != YoloTaskType::CLS || classes.empty()) {
            throw std::runtime_error("could not get top1 from YoloResult, "
                "it's not a classification task.");
        }
        return classes[0].cls_id;
    }

    float YoloResult::top1_conf() const {
        if (task != YoloTaskType::CLS || classes.empty()) {
            throw std::runtime_error("could not get top1 conf from YoloResult, "
                "it's not a classification task.");
        }
        return classes[0].conf;
    }

    std::string YoloResult::top1_label() const {
        if (task != YoloTaskType::CLS || classes.empty()) {
            throw std::runtime_error("could not get top1 label from YoloResult, "
                "it's not a classification task.");
        }
        return classes[0].label;
    }

    std::vector<int> YoloResult::top5() const {
        if (task != YoloTaskType::CLS || classes.empty()) {
            throw std::runtime_error("could not get top5 from YoloResult, "
                "it's not a classification task.");
        }
        // return the right number if size < 5
        auto n = std::min(classes.size(), static_cast<size_t>(5));
        std::vector<int> cls_ids;
        for (size_t i = 0; i < n; ++i) {
            cls_ids.push_back(classes[i].cls_id);
        }
        return cls_ids;
    }

    std::vector<float> YoloResult::top5_confs() const {
        if (task != YoloTaskType::CLS || classes.empty()) {
            throw std::runtime_error("could not get top5 confs from YoloResult, "
                "it's not a classification task.");
        }
        // return the right number if size < 5
        auto n = std::min(classes.size(), static_cast<size_t>(5));
        std::vector<float> confs;
        for (size_t i = 0; i < n; ++i) {
            confs.push_back(classes[i].conf);
        }
        return confs;
    }

    std::vector<std::string> YoloResult::top5_labels() const {
        if (task != YoloTaskType::CLS || classes.empty()) {
            throw std::runtime_error("could not get top5 labels from YoloResult, "
                "it's not a classification task.");
        }
        // return the right number if size < 5
        auto n = std::min(classes.size(), static_cast<size_t>(5));
        std::vector<std::string> labels;
        for (size_t i = 0; i < n; ++i) {
            labels.push_back(classes[i].label);
        }
        return labels;
    }

    std::vector<cv::Rect> YoloResult::boxes() const {
        if (task != YoloTaskType::DET &&
                task != YoloTaskType::SEG &&
                task != YoloTaskType::POSE) {
            throw std::runtime_error("could not get boxes from YoloResult, "
                "it's not a detection|segmentation|pose task.");
        }
        std::vector<cv::Rect> boxes;

        // priority: 
        // detection->segmentation->pose
        if (!detections.empty()) {
            boxes.reserve(detections.size());
            for (size_t i = 0; i < detections.size(); ++i) {
                boxes.emplace_back(detections[i].box);
            }
        }
        else if (!segmentations.empty()) {
            boxes.reserve(segmentations.size());
            for (size_t i = 0; i < segmentations.size(); ++i) {
                boxes.emplace_back(segmentations[i].box);
            }
        }
        else if (!poses.empty()) {
            boxes.reserve(poses.size());
            for (size_t i = 0; i < poses.size(); ++i) {
                boxes.emplace_back(poses[i].box);
            }
        }
        else {
            // should NOT throw error, 
            // maybe just no structured results output from Yolo.
        }
        return boxes;
    }

    std::vector<cv::RotatedRect> YoloResult::rboxes() const {
        if (task != YoloTaskType::OBB) {
            throw std::runtime_error("could not get rboxes from YoloResult, "
                "it's not a obb task.");
        }

        std::vector<cv::RotatedRect> rboxes;
        if (!obbs.empty()) {
            rboxes.reserve(obbs.size());
            for (size_t i = 0; i < obbs.size(); ++i) {
                rboxes.emplace_back(obbs[i].rbox);
            }
        }
        else {
            // should NOT throw error, 
            // maybe just no structured results output from Yolo.
        }
        return rboxes;
    }

    std::vector<int> YoloResult::cls_ids() const {
        if (task != YoloTaskType::DET &&
                task != YoloTaskType::SEG &&
                task != YoloTaskType::POSE &&
                task != YoloTaskType::OBB) {
            throw std::runtime_error("could not get cls_ids from YoloResult, "
                "it's not a detection|segmentation|pose|obb task.");
        }
        std::vector<int> cls_ids;

        // priority: 
        // detection->segmentation->pose->obb
        if (!detections.empty()) {
            cls_ids.reserve(detections.size());
            for (size_t i = 0; i < detections.size(); ++i) {
                cls_ids.emplace_back(detections[i].cls_id);
            }
        }
        else if (!segmentations.empty()) {
            cls_ids.reserve(segmentations.size());
            for (size_t i = 0; i < segmentations.size(); ++i) {
                cls_ids.emplace_back(segmentations[i].cls_id);
            }
        }
        else if (!poses.empty()) {
            cls_ids.reserve(poses.size());
            for (size_t i = 0; i < poses.size(); ++i) {
                cls_ids.emplace_back(poses[i].cls_id);
            }
        }
        else if (!obbs.empty()) {
            cls_ids.reserve(obbs.size());
            for (size_t i = 0; i < obbs.size(); ++i) {
                cls_ids.emplace_back(obbs[i].cls_id);
            }
        }
        else {
            // should NOT throw error, 
            // maybe just no structured results output from Yolo.
        }
        return cls_ids;
    }

    std::vector<float> YoloResult::confs() const {
        if (task != YoloTaskType::DET &&
                task != YoloTaskType::SEG &&
                task != YoloTaskType::POSE &&
                task != YoloTaskType::OBB) {
            throw std::runtime_error("could not get confs from YoloResult, "
                "it's not a detection|segmentation|pose|obb task.");
        }
        std::vector<float> confs;

        // priority: 
        // detection->segmentation->pose->obb
        if (!detections.empty()) {
            confs.reserve(detections.size());
            for (size_t i = 0; i < detections.size(); ++i) {
                confs.emplace_back(detections[i].conf);
            }
        }
        else if (!segmentations.empty()) {
            confs.reserve(segmentations.size());
            for (size_t i = 0; i < segmentations.size(); ++i) {
                confs.emplace_back(segmentations[i].conf);
            }
        }
        else if (!poses.empty()) {
            confs.reserve(poses.size());
            for (size_t i = 0; i < poses.size(); ++i) {
                confs.emplace_back(poses[i].conf);
            }
        }
        else if (!obbs.empty()) {
            confs.reserve(obbs.size());
            for (size_t i = 0; i < obbs.size(); ++i) {
                confs.emplace_back(obbs[i].conf);
            }
        }
        else {
            // should NOT throw error, 
            // maybe just no structured results output from Yolo.
        }
        return confs;
    }

    std::vector<std::string> YoloResult::labels() const {
        if (task != YoloTaskType::DET &&
                task != YoloTaskType::SEG &&
                task != YoloTaskType::POSE &&
                task != YoloTaskType::OBB) {
            throw std::runtime_error("could not get labels from YoloResult, "
                "it's not a detection|segmentation|pose|obb task.");
        }
        std::vector<std::string> labels;

        // priority: 
        // detection->segmentation->pose->obb
        if (!detections.empty()) {
            labels.reserve(detections.size());
            for (size_t i = 0; i < detections.size(); ++i) {
                labels.emplace_back(detections[i].label);
            }
        }
        else if (!segmentations.empty()) {
            labels.reserve(segmentations.size());
            for (size_t i = 0; i < segmentations.size(); ++i) {
                labels.emplace_back(segmentations[i].label);
            }
        }
        else if (!poses.empty()) {
            labels.reserve(poses.size());
            for (size_t i = 0; i < poses.size(); ++i) {
                labels.emplace_back(poses[i].label);
            }
        }
        else if (!obbs.empty()) {
            labels.reserve(obbs.size());
            for (size_t i = 0; i < obbs.size(); ++i) {
                labels.emplace_back(obbs[i].label);
            }
        }
        else {
            // should NOT throw error, 
            // maybe just no structured results output from Yolo.
        }
        return labels;
    }

    std::vector<cv::Mat> YoloResult::masks() const {
        if (task != YoloTaskType::SEG) {
            throw std::runtime_error("could not get masks from YoloResult, "
                "it's not a segmentation task.");
        }

        std::vector<cv::Mat> masks;
        if (!segmentations.empty()) {
            masks.reserve(segmentations.size());
            for (size_t i = 0; i < segmentations.size(); ++i) {
                masks.emplace_back(segmentations[i].mask);
            }
        }
        else {
            // should NOT throw error, 
            // maybe just no structured results output from Yolo.
        }
        return masks;
    }

    std::vector<std::vector<cv::Point>> YoloResult::contours() const {
        if (task != YoloTaskType::SEG) {
            throw std::runtime_error("could not get contours from YoloResult, "
                "it's not a segmentation task.");
        }

        std::vector<std::vector<cv::Point>> contours;
        if (!segmentations.empty()) {
            contours.reserve(segmentations.size());
            for (size_t i = 0; i < segmentations.size(); ++i) {
                contours.emplace_back(segmentations[i].contour);
            }
        }
        else {
            // should NOT throw error, 
            // maybe just no structured results output from Yolo.
        }
        return contours;
    }

    std::vector<std::vector<YoloKeyPoint>> YoloResult::kpts() const {
        if (task != YoloTaskType::POSE) {
            throw std::runtime_error("could not get kpts from YoloResult, "
                "it's not a pose task.");
        }

        std::vector<std::vector<YoloKeyPoint>> kpts;
        if (!poses.empty()) {
            kpts.reserve(poses.size());
            for (size_t i = 0; i < poses.size(); ++i) {
                kpts.emplace_back(poses[i].keypoints);
            }
        }
        else {
            // should NOT throw error, 
            // maybe just no structured results output from Yolo.
        }
        return kpts;
    }

    std::vector<int> YoloResult::track_ids() const {
        if (task != YoloTaskType::DET &&
                task != YoloTaskType::SEG &&
                task != YoloTaskType::POSE) {
            throw std::runtime_error("could not get track ids from YoloResult, "
                "it's not a detection|segmentation|pose task.");
        }
        std::vector<int> track_ids;

        // priority: 
        // detection->segmentation->pose
        if (!detections.empty()) {
            track_ids.reserve(detections.size());
            for (size_t i = 0; i < detections.size(); ++i) {
                track_ids.emplace_back(detections[i].track_id);
            }
        }
        else if (!segmentations.empty()) {
            track_ids.reserve(segmentations.size());
            for (size_t i = 0; i < segmentations.size(); ++i) {
                track_ids.emplace_back(segmentations[i].track_id);
            }
        }
        else if (!poses.empty()) {
            track_ids.reserve(poses.size());
            for (size_t i = 0; i < poses.size(); ++i) {
                track_ids.emplace_back(poses[i].track_id);
            }
        }
        else {
            // should NOT throw error, 
            // maybe just no structured results output from Yolo.
        }
        return track_ids;     
    }

    std::vector<std::vector<cv::Point>> YoloResult::track_points() const {
        if (task != YoloTaskType::DET &&
                task != YoloTaskType::SEG &&
                task != YoloTaskType::POSE) {
            throw std::runtime_error("could not get track points from YoloResult, "
                "it's not a detection|segmentation|pose task.");
        }
        std::vector<std::vector<cv::Point>> track_points;

        // priority: 
        // detection->segmentation->pose
        if (!detections.empty()) {
            track_points.reserve(detections.size());
            for (size_t i = 0; i < detections.size(); ++i) {
                track_points.emplace_back(detections[i].track_points);
            }
        }
        else if (!segmentations.empty()) {
            track_points.reserve(segmentations.size());
            for (size_t i = 0; i < segmentations.size(); ++i) {
                track_points.emplace_back(segmentations[i].track_points);
            }
        }
        else if (!poses.empty()) {
            track_points.reserve(poses.size());
            for (size_t i = 0; i < poses.size(); ++i) {
                track_points.emplace_back(poses[i].track_points);
            }
        }
        else {
            // should NOT throw error, 
            // maybe just no structured results output from Yolo.
        }
        return track_points;  
    }
}