#include "YoloObjs.h"

namespace yolo {
    void to_json(json& j, const YoloClsObj& obj) {
        j = json{
            {"cls_id", obj.cls_id}, 
            {"conf", obj.conf}, 
            {"label", obj.label}};
    }

    void to_json(json& j, const YoloDetObj& obj) {
        j = json{
            {"box", obj.box}, 
            {"cls_id", obj.cls_id}, 
            {"conf", obj.conf}, 
            {"label", obj.label},
            {"track_id", obj.track_id}};
    }

    void to_json(json& j, const YoloSegObj& obj) {
        j = json{
            {"box", obj.box}, 
            {"cls_id", obj.cls_id}, 
            {"conf", obj.conf}, 
            {"label", obj.label},
            {"track_id", obj.track_id}};
    }

    void to_json(json& j, const YoloPoseObj& obj) {
        j = json{
            {"box", obj.box}, 
            {"cls_id", obj.cls_id}, 
            {"conf", obj.conf}, 
            {"label", obj.label},
            {"track_id", obj.track_id},
            {"keypoints", obj.keypoints}};
    }

    void to_json(json& j, const YoloObbObj& obj) {
        j = json{
            {"rbox", obj.rbox}, 
            {"cls_id", obj.cls_id}, 
            {"conf", obj.conf}, 
            {"label", obj.label}};
    }

    void to_json(json& j, const YoloKeyPoint& obj) {
        j = json{
            {"x", obj.x}, 
            {"y", obj.y}, 
            {"conf", obj.conf}};
    }
}

namespace cv {
    void to_json(json& j, const cv::Point& obj) {
        j = json{
            {"x", obj.x}, 
            {"y", obj.y}};
    }

    void to_json(json& j, const cv::Rect& obj) {
        j = json{
            {"x", obj.x}, 
            {"y", obj.y}, 
            {"width", obj.width},
            {"height", obj.height}};
    }

    void to_json(json& j, const cv::RotatedRect& obj) {
        j = json{
            {"cx", obj.center.x}, 
            {"cy", obj.center.y}, 
            {"width", obj.size.width},
            {"height", obj.size.height},
            {"angle", obj.angle}};
    }
}