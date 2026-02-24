#include "track/YoloTracker.h"
#include "track/sort/SortTrackAlgo.h"

namespace yolo {
    YoloTracker::YoloTracker(
        const YoloTrackConfig& cfg): __cfg(cfg) {
        init();
    }

    YoloTracker::~YoloTracker() {

    }

    void YoloTracker::init() {
        // choose track algorithm
        switch (__cfg.algo) {
        case YoloTrackAlgo::SORT: {
            __tracker = std::make_shared<SortTrackAlgo>(__cfg);
            break;
        }
        case YoloTrackAlgo::BYTE_TRACK: {
            throw std::invalid_argument("invalid YoloTrackAlgo parameter when initializing YoloTracker!");
            break;
        }
        default:
            throw std::invalid_argument("invalid YoloTrackAlgo parameter when initializing YoloTracker!");
            break;
        }

        // initialize status
        __tracking_points.clear();
        __tracking_miss_times.clear();
    }

    void YoloTracker::reset() {
        init(); 
    }

    void YoloTracker::preprocess(
        const YoloResult& res, 
        std::vector<cv::Rect>& boxes, 
        std::vector<std::vector<float>>& embeddings
    ) {
        auto res_boxes = res.boxes();
        boxes.insert(boxes.end(), res_boxes.begin(), res_boxes.end());

        /* embeddings reserved because no embeddings in YoloResult now */
        // auto res_embeddings = res.embeddings();
        // embeddings.insert(embeddings.end(), res_embeddings.begin(), res_embeddings.end());
    }

    void YoloTracker::run(
        const std::vector<cv::Rect>& boxes, 
        const std::vector<std::vector<float>>& embeddings, 
        std::vector<int>& track_ids
    ) {
        (*__tracker).run(boxes, embeddings, track_ids);
    }

    void YoloTracker::postprocess(
        const std::vector<cv::Rect>& boxes, 
        const std::vector<std::vector<float>>& embeddings, 
        const std::vector<int>& track_ids,
        YoloResult& res
    ) {
        assert(boxes.size() == track_ids.size());
        // assert(embeddings.size() == track_ids.size());

        for (size_t i = 0; i < track_ids.size(); i++) {
            auto& box      = boxes[i];
            auto& track_id = track_ids[i];

            if (track_id < 0) {
                continue;
            }
            
            cv::Point track_point;

            switch (__cfg.loc) {
            case YoloTrackLoc::CENTER: {
                track_point.x = box.x + box.width / 2;
                track_point.y = box.y + box.height / 2;
                break;
            }
            case YoloTrackLoc::BOTTOM_CENTER: {
                track_point.x = box.x + box.width / 2;
                track_point.y = box.y + box.height;
                break;
            }
            case YoloTrackLoc::BOTTOM_CUSTOM: {
                track_point.x = box.x + int(box.width * __cfg.loc_f);
                track_point.y = box.y + box.height;
                break;
            }
            default:
                throw std::runtime_error("got unsupported YoloTrackLoc when calling YoloTracker::postprocess()!");
                break;
            }

            __tracking_points[track_id].emplace_back(track_point);
            // reset to 0 since it got hit
            __tracking_miss_times[track_id] = 0;

            /* update track id & track points for YoloResult via indice directly
               important: they have the same indice order
            */
            switch (res.task) {
            case YoloTaskType::DET: {
                res.detections[i].track_id = track_id;
                res.detections[i].track_points = __tracking_points[track_id];
                break;
            }
            case YoloTaskType::SEG: {
                res.segmentations[i].track_id = track_id;
                res.segmentations[i].track_points = __tracking_points[track_id];
                break;
            }
            case YoloTaskType::POSE: {
                res.poses[i].track_id = track_id;
                res.poses[i].track_points = __tracking_points[track_id];
                break;
            }
            default:
                throw std::runtime_error("got unsupported YoloTaskType when calling YoloTracker::postprocess()!");
                break;
            }
        }

        // check miss times & clear garbage data
        for (auto i = __tracking_miss_times.begin(); i != __tracking_miss_times.end();) {
            // not got hit, increase by 1
            if (i->second) {
                i->second++;
            }

            if (i->second > __cfg.max_miss) {
                __tracking_points.erase(i->first);
                i = __tracking_miss_times.erase(i);
            }
            else {
                i++;
            }

            //assert(__tracking_miss_times.size() == __tracking_points.size()); 
        }
    }

    void YoloTracker::track(YoloResult& res) {
        if (res.task != YoloTaskType::DET
            && res.task != YoloTaskType::SEG
            && res.task != YoloTaskType::POSE
        ) {
            throw std::runtime_error("got unsupported task type when calling YoloTracker::track()!");
            return;
        }
        // support bbox & embedding as input
        std::vector<cv::Rect>           boxes;
        std::vector<std::vector<float>> embeddings;
        std::vector<int>                track_ids;
        // step1. preprocess, collect boxes & embeddings(Reserved)
        preprocess(res, boxes, embeddings);

        // step2. run track algorithm
        run(boxes, embeddings, track_ids);

        // step3. postprocess, update YoloResult
        postprocess(boxes, embeddings, track_ids, res);
    }

    YoloResult YoloTracker::track_copy(const YoloResult& res) {
        auto copy = res;
        track(copy);

        return copy;
    }

    void YoloTracker::operator()(YoloResult& res) {
        track(res);
    }

    std::string YoloTracker::info(bool print) {
        return "";
    }
}