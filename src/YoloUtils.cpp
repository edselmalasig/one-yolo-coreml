#include "YoloUtils.h"

namespace yolo {
    std::string to_string(const float f, const int precision) {
        std::ostringstream out;
        out.precision(precision);
        out << std::fixed << f;
        return out.str();
    }

    cv::Mat draw_results(
        const cv::Mat&                  image,
        const DrawParam&                param,
        const std::vector<int>&         top5,
        const std::vector<float>&       top5_confs,
        const std::vector<std::string>& top5_labels,
        const std::vector<int>&         cls_ids,
        const std::vector<float>&       confs,
        const std::vector<std::string>& labels,
        const std::vector<cv::Rect>&    boxes,
        const std::vector<cv::RotatedRect>&           rboxes,
        const std::vector<cv::Mat>&                   masks,
        const std::vector<std::vector<cv::Point>>&    contours,
        const std::vector<std::vector<YoloKeyPoint>>& kpts,
        const std::vector<int>&                       track_ids,
        const std::vector<std::vector<cv::Point>>&    tracks) {
        auto canvas = image.clone();
        // resize or not

        auto colors     = get_colors_48();
        auto colors_num = colors.size();
        auto font_face  = param.font_face;
        auto font_scale = param.font_scale;
        auto font_color = param.font_color;
        auto font_thickness  = param.font_thickness;
        auto font_offset_x   = param.box_line_width / 2;
        auto font_padding    = 8;

        // classification task
        if (!top5.empty()) {
            assert(top5.size() == top5_confs.size());
            assert(top5.size() == top5_labels.size());

            auto num    = param.top1_only ? 1 : top5.size();
            auto loc = cv::Point(font_padding, font_padding);  // start point
            for (size_t i = 0; i < num; i++) {
                auto color = colors[top5[i] % colors_num];
                // [rank]: cls_id, conf, label
                std::string txt = "[rank" + std::to_string(i + 1) + "]: " + (param.cls_ids ? std::to_string(top5[i]) + ", " : "") 
                    + to_string(top5_confs[i] * 100) + "%, " + top5_labels[i];
                // calculate rectangle of txt
                auto baseline = 0;
                auto txt_size = cv::getTextSize(
                    txt, font_face, font_scale, 
                    font_thickness, &baseline);
                cv::rectangle(
                    canvas, 
                    cv::Rect(
                        loc.x, 
                        loc.y, 
                        txt_size.width + font_padding * 2, 
                        txt_size.height + baseline + font_padding * 2), 
                    color, -1);  // fill
                cv::putText(
                    canvas, txt, 
                    cv::Point(
                        loc.x + font_padding, 
                        loc.y + txt_size.height + font_padding), 
                    font_face, font_scale, font_color, font_thickness);
                loc = cv::Point(
                    loc.x, 
                    loc.y + txt_size.height + baseline + font_padding * 2 + font_padding);   // add padding space in vertical
            }
        }

        // detection/segmentation/pose tasks
        if (!tracks.empty() && param.tracks && param.track_line_width > 0) {
            assert(boxes.size() == cls_ids.size());
            assert(boxes.size() == confs.size());
            assert(boxes.size() == labels.size());
            assert(boxes.size() == tracks.size());

            for (size_t i = 0; i < tracks.size(); i++) {
                auto& one_track = tracks[i];
                auto color = param.color_by_class ? colors[cls_ids[i] % colors_num] : colors[i % colors_num];

                for (size_t j = 0; j + 1 < one_track.size(); j++) {
                    auto& p1 = one_track[j];
                    auto& p2 = one_track[j + 1];
                    cv::line(
                        canvas, p1, p2, 
                        color, param.track_line_width, cv::LINE_AA);
                    
                    // the last one is located point
                    if (j + 2 == one_track.size() && param.loc_radius > 0) {
                        cv::circle(canvas, p2, param.loc_radius, color, -1);
                    }
                }
            }
        }
        
        // segmentation task
        if (!contours.empty() && !masks.empty() && param.masks) {
            assert(boxes.size() == cls_ids.size());
            assert(boxes.size() == confs.size());
            assert(boxes.size() == labels.size());
            assert(boxes.size() == masks.size());
            assert(boxes.size() == contours.size());

            // we use contours not use masks here
            for (size_t i = 0; i < contours.size(); i++) {
                auto color = param.color_by_class ? colors[cls_ids[i] % colors_num] : colors[i % colors_num];

                auto one_contour  = contours[i];
                // convert to local point
                for (auto& p: one_contour) {
                    p.x -= boxes[i].x;
                    p.y -= boxes[i].y;
                }
                float mask_alpha = std::max(0.0f, std::min(1.0f, param.mask_alpha));
                cv::Mat roi      = canvas(boxes[i]);  // reference
                cv::Mat overlay  = roi.clone();
                // fill contour
                cv::drawContours(
                    overlay,
                    std::vector<std::vector<cv::Point>>{one_contour},
                    -1,
                    color,
                    cv::FILLED
                );

                // roi = mask_alpha * overlay + (1.0 - mask_alpha) * roi
                cv::addWeighted(overlay, mask_alpha,
                                roi,     1.0f - mask_alpha,
                                0.0,
                                roi,
                                -1);
                // draw contour
                if (param.mask_line_width > 0) {
                    cv::drawContours(
                        roi,
                        std::vector<std::vector<cv::Point>>{one_contour},
                        -1,
                        color,
                        param.mask_line_width
                    );  
                }
            }
            
            /*
            for (size_t i = 0; i < masks.size(); i++) {
                auto color = param.color_by_class ? colors[cls_ids[i] % colors_num] : colors[i % colors_num];

                // binary mask (0 background / 255 object), has the same size as bbox
                auto& mask = masks[i];

                float mask_alpha = std::max(0.0f, std::min(1.0f, param.mask_alpha));
                cv::Mat roi = canvas(boxes[i]);  // reference
                cv::Mat color_mat(roi.size(), roi.type(), color);

                // blended = mask_alpha * color_mat + (1.0 - mask_alpha) * roi
                cv::Mat blended;
                cv::addWeighted(color_mat, mask_alpha,
                                roi,       1.0f - mask_alpha,
                                0.0,
                                blended,
                                -1);

                // draw contour of mask
                if (param.mask_line_width > 0) {
                    std::vector<cv::Mat> contours;
                    cv::Mat hierarchy;
                    cv::findContours(mask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
                    cv::drawContours(blended, contours, -1, color, param.mask_line_width, cv::LINE_AA, hierarchy, 100);
                }
                blended.copyTo(roi, mask);
            }*/
        }

        // pose task
        if (!kpts.empty() && param.kpts) {
            assert(boxes.size() == cls_ids.size());
            assert(boxes.size() == confs.size());
            assert(boxes.size() == labels.size());
            assert(boxes.size() == kpts.size());

            for (size_t i = 0; i < kpts.size(); i++) {
                auto& one_kpts = kpts[i]; 

                // draw link lines between points
                if (param.kpt_line_width > 0) {
                    // make sure each line part has itself color
                    assert(param.kpt_pairs.size() == param.kpt_pair_colors.size());

                    for (size_t j = 0; j < param.kpt_pairs.size(); j++) {
                        auto pair_indices = param.kpt_pairs[j];
                        auto pair_color   = param.kpt_pair_colors[j];

                        // conf of keypoint meet the threshold
                        if (one_kpts[pair_indices.first].conf >= 0.5 &&
                            one_kpts[pair_indices.second].conf >= 0.5) {
                            auto p1 = cv::Point(
                                one_kpts[pair_indices.first].x,
                                one_kpts[pair_indices.first].y);
                            auto p2 = cv::Point(
                                one_kpts[pair_indices.second].x, 
                                one_kpts[pair_indices.second].y);
                            cv::line(
                                canvas, p1, p2, 
                                pair_color, param.kpt_line_width, cv::LINE_AA);
                        }
                    }
                }
                
                // draw points
                if (param.kpt_radius > 0) {
                    auto color = param.color_by_class ? colors[cls_ids[i] % colors_num] : colors[i % colors_num];
                    for (size_t j = 0; j < one_kpts.size(); j++) {
                        if (one_kpts[j].conf >= 0.5) {
                            auto p = cv::Point(one_kpts[j].x, one_kpts[j].y);
                            cv::circle(
                                canvas, p, 
                                param.kpt_radius, color, -1);  // fill
                        }
                    }
                    
                }
            }
        }

        // detection/segmentation/pose tasks
        if (!boxes.empty() && param.boxes && param.box_line_width > 0) {
            assert(boxes.size() == cls_ids.size());
            assert(boxes.size() == confs.size());
            assert(boxes.size() == labels.size());
            assert(boxes.size() == track_ids.size());

            for (size_t i = 0; i < boxes.size(); i++) {
                auto color = param.color_by_class ? colors[cls_ids[i] % colors_num] : colors[i % colors_num];
                cv::rectangle(canvas, boxes[i], color, param.box_line_width);

                std::string txt = "";
                if (param.track_ids && track_ids[i] >= 0) {
                    // track_id == -1 means no tracked, ignored
                    txt += "#" + std::to_string(track_ids[i]);
                }
                if (param.cls_ids) {
                    txt += (!txt.empty() ? ", " : "") + std::to_string(cls_ids[i]);
                }
                if (param.labels) {
                    txt += (!txt.empty() ? ", " : "") + labels[i];
                }
                if (param.confs) {
                    txt += (!txt.empty() ? ", " : "") + to_string(confs[i] * 100) + "%";
                }

                // draw text
                if (!txt.empty()) {
                    // calculate rectangle of txt
                    auto baseline = 0;
                    auto txt_size = cv::getTextSize(
                        txt, 
                        font_face, font_scale, 
                        font_thickness, &baseline);
                    cv::rectangle(
                        canvas, 
                        cv::Rect(
                            boxes[i].x - font_offset_x, 
                            boxes[i].y - txt_size.height - baseline - font_padding * 2, 
                            txt_size.width + font_padding * 2, 
                            txt_size.height + baseline + font_padding * 2), 
                        color, -1);  // fill
                    cv::putText(
                        canvas, txt, 
                        cv::Point(
                            boxes[i].x - font_offset_x + font_padding, 
                            boxes[i].y - baseline - font_padding), 
                        font_face, font_scale, 
                        font_color, font_thickness);
                }
            }
        }

        // obb task
        if (!rboxes.empty() && param.rboxes && param.rbox_line_width > 0) {
            assert(rboxes.size() == cls_ids.size());
            assert(rboxes.size() == confs.size());
            assert(rboxes.size() == labels.size());

            for (size_t i = 0; i < rboxes.size(); i++) {
                auto color = param.color_by_class ? colors[cls_ids[i] % colors_num] : colors[i % colors_num];

                // find 4 vertices for rotated rect first and then draw polylines
                std::vector<cv::Point2f> vertices_f;
                rboxes[i].points(vertices_f);
                std::vector<cv::Point> i_vertices;
                for (int i = 0; i < 4; i++)
                    i_vertices.push_back(cv::Point(cvRound(vertices_f[i].x), cvRound(vertices_f[i].y)));
                cv::polylines(
                    canvas, i_vertices, 
                    true, color, param.rbox_line_width, cv::LINE_AA);

                std::string txt = "";
                if (param.cls_ids) {
                    txt += std::to_string(cls_ids[i]);
                }
                if (param.labels) {
                    txt += (!txt.empty() ? ", " : "") + labels[i];
                }
                if (param.confs) {
                    txt += (!txt.empty() ? ", " : "") + to_string(confs[i] * 100) + "%";
                }

                // draw text
                if (!txt.empty()) {
                    // calculate rectangle of txt
                    auto baseline = 0;
                    auto txt_size = cv::getTextSize(
                        txt, 
                        font_face, font_scale, 
                        font_thickness, &baseline);
                    cv::rectangle(
                        canvas, 
                        cv::Rect(
                            i_vertices[1].x - font_offset_x, 
                            i_vertices[1].y - txt_size.height - baseline - font_padding * 2, 
                            txt_size.width + font_padding * 2, 
                            txt_size.height + baseline + font_padding * 2), 
                        color, -1);  // fill
                    cv::putText(
                        canvas, txt, 
                        cv::Point(
                            i_vertices[1].x - font_offset_x + font_padding, 
                            i_vertices[1].y - baseline - font_padding), 
                        font_face, font_scale, 
                        font_color, font_thickness);
                }
            }
        }

        return canvas;
    }

    std::vector<cv::Scalar> get_colors_48() {
        return {
            cv::Scalar(0, 100, 0),       // DarkGreen
            cv::Scalar(0, 0, 139),       // DarkRed
            cv::Scalar(139, 0, 0),       // DarkBlue
            cv::Scalar(139, 0, 139),     // DarkMagenta
            cv::Scalar(0, 139, 139),     // Olive (dark)
            cv::Scalar(139, 139, 0),     // Teal (dark)
            cv::Scalar(130, 0, 75),      // Indigo
            cv::Scalar(128, 0, 128),     // Purple
            cv::Scalar(240, 32, 160),    // DarkViolet
            cv::Scalar(211, 0, 148),     // DarkOrchid
            cv::Scalar(130, 0, 75),      // Indigo
            cv::Scalar(19, 69, 139),     // SaddleBrown
            cv::Scalar(45, 82, 160),     // Sienna
            cv::Scalar(33, 67, 101),     // DarkBrown
            cv::Scalar(47, 107, 85),     // DarkOliveGreen
            cv::Scalar(87, 139, 46),     // SeaGreen
            cv::Scalar(128, 128, 0),     // Teal
            cv::Scalar(150, 75, 0),      // Deep Blue
            cv::Scalar(75, 0, 150),      // Deep Magenta
            cv::Scalar(0, 150, 75),      // Deep Green
            cv::Scalar(0, 50, 100),      // Dark Orange-Brown
            cv::Scalar(100, 50, 0),      // Navy Green
            cv::Scalar(100, 0, 50),      // Deep Purple-Blue
            cv::Scalar(50, 0, 100),      // Maroon-ish
            cv::Scalar(0, 100, 50),      // Forest Green
            cv::Scalar(50, 100, 0),      // Green-Teal
            cv::Scalar(100, 50, 100),    // Plum-like
            cv::Scalar(100, 100, 50),    // Dark Cyan
            cv::Scalar(80, 40, 120),     // Wine
            cv::Scalar(120, 40, 80),     // Eggplant
            cv::Scalar(120, 80, 40),     // Steel Blue
            cv::Scalar(40, 80, 120),     // Mustard Brown
            cv::Scalar(60, 30, 110),     // Burgundy
            cv::Scalar(110, 30, 60),     // Royal Purple
            cv::Scalar(110, 60, 30),     // Ocean Blue
            cv::Scalar(30, 60, 110),     // Rust
            cv::Scalar(30, 110, 60),     // Lime Forest
            cv::Scalar(60, 110, 30),     // Jungle Green
            cv::Scalar(70, 20, 90),      // Deep Rose
            cv::Scalar(90, 20, 70),      // Velvet
            cv::Scalar(90, 70, 20),      // Deep Sky
            cv::Scalar(20, 70, 90),      // Bronze
            cv::Scalar(50, 25, 100),     // Crimson Dark
            cv::Scalar(100, 25, 50),     // Midnight Purple
            cv::Scalar(80, 120, 40),     // Moss Green
            cv::Scalar(40, 120, 80),     // Olive Drab
            cv::Scalar(20, 90, 70),      // Avocado
            cv::Scalar(70, 90, 20)       // Peacock
        };
    }

    YoloUtils::YoloUtils(/* args */)
    {
    }
    
    YoloUtils::~YoloUtils()
    {
    }

    cv::Mat YoloUtils::letterbox(
        const cv::Mat& img,
        int new_w,
        int new_h,
        LetterBoxInfo& info,
        const cv::Scalar& color
    ) {
        int w = img.cols;
        int h = img.rows;

        float r = std::min(
            static_cast<float>(new_w) / w,
            static_cast<float>(new_h) / h
        );

        int resized_w = std::round(w * r);
        int resized_h = std::round(h * r);

        cv::Mat resized;
        cv::resize(img, resized, cv::Size(resized_w, resized_h));

        int pad_w = new_w - resized_w;
        int pad_h = new_h - resized_h;

        int pad_left   = pad_w / 2;
        int pad_right  = pad_w - pad_left;
        int pad_top    = pad_h / 2;
        int pad_bottom = pad_h - pad_top;

        cv::Mat padded;
        cv::copyMakeBorder(
            resized, padded,
            pad_top, pad_bottom,
            pad_left, pad_right,
            cv::BORDER_CONSTANT,
            color
        );

        info.scale = r;
        info.pad_w = pad_left;
        info.pad_h = pad_top;

        return padded;
    }

    void YoloUtils::class_aware_nms(
        const std::vector<cv::Rect>& boxes,
        const std::vector<float>& scores,
        const std::vector<int>& cls_ids,
        float conf_thresh,
        float nms_thresh,
        std::vector<int>& keep_indices
    ) {
        keep_indices.clear();

        // class_id -> indices
        std::unordered_map<int, std::vector<int>> cls_map;
        for (int i = 0; i < (int)cls_ids.size(); ++i) {
            cls_map[cls_ids[i]].push_back(i);
        }

        // per-class NMS
        for (const auto& kv : cls_map) {
            const auto& indices = kv.second;

            std::vector<cv::Rect> cls_boxes;
            std::vector<float>    cls_scores;

            cls_boxes.reserve(indices.size());
            cls_scores.reserve(indices.size());

            for (int idx : indices) {
                cls_boxes.emplace_back(boxes[idx]);
                cls_scores.emplace_back(scores[idx]);
            }

            std::vector<int> cls_keep;
            cv::dnn::NMSBoxes(
                cls_boxes,
                cls_scores,
                conf_thresh,
                nms_thresh,
                cls_keep
            );

            // 映射回原索引
            for (int k : cls_keep) {
                keep_indices.push_back(indices[k]);
            }
        }
    }

    void YoloUtils::class_aware_nms(
        const std::vector<cv::RotatedRect>& rboxes,
        const std::vector<float>& scores,
        const std::vector<int>& cls_ids,
        float conf_thresh,
        float nms_thresh,
        std::vector<int>& keep_indices
    ) {
        keep_indices.clear();

        // class_id -> indices
        std::unordered_map<int, std::vector<int>> cls_map;
        for (int i = 0; i < (int)cls_ids.size(); ++i) {
            cls_map[cls_ids[i]].push_back(i);
        }

        // per-class NMS
        for (const auto& kv : cls_map) {
            const auto& indices = kv.second;

            std::vector<cv::RotatedRect> cls_rboxes;
            std::vector<float>    cls_scores;

            cls_rboxes.reserve(indices.size());
            cls_scores.reserve(indices.size());

            for (int idx : indices) {
                cls_rboxes.emplace_back(rboxes[idx]);
                cls_scores.emplace_back(scores[idx]);
            }

            std::vector<int> cls_keep;
            cv::dnn::NMSBoxes(
                cls_rboxes,
                cls_scores,
                conf_thresh,
                nms_thresh,
                cls_keep
            );

            // 映射回原索引
            for (int k : cls_keep) {
                keep_indices.push_back(indices[k]);
            }
        }
    }

    cv::Rect YoloUtils::decode_box(
        float cx, float cy, float w, float h,
        const LetterBoxInfo& lb,
        const cv::Size& orig_size
    ) {
        float x1 = cx - w * 0.5f;
        float y1 = cy - h * 0.5f;
        float x2 = cx + w * 0.5f;
        float y2 = cy + h * 0.5f;

        x1 = (x1 - lb.pad_w) / lb.scale;
        y1 = (y1 - lb.pad_h) / lb.scale;
        x2 = (x2 - lb.pad_w) / lb.scale;
        y2 = (y2 - lb.pad_h) / lb.scale;

        x1 = std::clamp(x1, 0.f, (float)orig_size.width  - 1.f);
        y1 = std::clamp(y1, 0.f, (float)orig_size.height - 1.f);
        x2 = std::clamp(x2, 0.f, (float)orig_size.width  - 1.f);
        y2 = std::clamp(y2, 0.f, (float)orig_size.height - 1.f);

        return cv::Rect(
            (int)x1,
            (int)y1,
            (int)(x2 - x1),
            (int)(y2 - y1)
        );
    }

    YoloKeyPoint YoloUtils::decode_keypoint(
        float x, float y, float conf,
        const LetterBoxInfo& lb,
        const cv::Size& orig_size
    ) {
        x = (x - lb.pad_w) / lb.scale;
        y = (y - lb.pad_h) / lb.scale;

        x = std::clamp(x, 0.f, (float)orig_size.width  - 1.f);
        y = std::clamp(y, 0.f, (float)orig_size.height - 1.f);

        return YoloKeyPoint {x, y, conf};
    }

    cv::RotatedRect YoloUtils::decode_rbox(
        float cx, float cy, float w, float h, float angle,
        const LetterBoxInfo& lb,
        const cv::Size& orig_size
    ) {
        cx = (cx - lb.pad_w) / lb.scale;
        cy = (cy - lb.pad_h) / lb.scale;

        w = w / lb.scale;
        h = h / lb.scale;

        cx = std::clamp(cx, 0.f, (float)orig_size.width  - 1.f);
        cy = std::clamp(cy, 0.f, (float)orig_size.height - 1.f);

        w = std::clamp(w, 0.f, (float)orig_size.width  - 1.f);
        h = std::clamp(h, 0.f, (float)orig_size.height - 1.f);

        float angle_deg = angle * 180.0f / CV_PI;
        return cv::RotatedRect(cv::Point(cx, cy), cv::Size(w, h), angle_deg);
    }
}