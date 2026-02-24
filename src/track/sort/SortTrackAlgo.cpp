#include "track/sort/SortTrackAlgo.h"

namespace yolo {
    
    SortTrackAlgo::SortTrackAlgo(const YoloTrackConfig& cfg): BaseTrackAlgo(cfg) {

    }
    
    SortTrackAlgo::~SortTrackAlgo() {

    }

    double SortTrackAlgo::getIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt) {
        float in = (bb_test & bb_gt).area();
        float un = bb_test.area() + bb_gt.area() - in;

        if (un < DBL_EPSILON)
            return 0;

        return (double)(in / un);
    }

    void SortTrackAlgo::run(
        const std::vector<cv::Rect>& boxes, 
        const std::vector<std::vector<float>>& embeddings, 
        std::vector<int>& track_ids
    ) {
        // fill track_ids with default value(-1) according to boxes' size (embeddings ignored)
		track_ids.resize(boxes.size());
		for (auto& track_id : track_ids) {
			track_id = -1;
		}

        // first time to initialize KalmanTracker
        if (trackers.empty()) {
            for (unsigned int i = 0; i < boxes.size(); i++) {
				auto trk = KalmanTracker(
                    cv::Rect_<float>(
                        boxes[i].x, 
                        boxes[i].y, 
                        boxes[i].width, 
                        boxes[i].height));
				trackers.emplace_back(trk);
			}
            return;
        }

        //3.1. get predicted locations from existing trackers.
        predictedBoxes.clear();
		for (auto it = trackers.begin(); it != trackers.end();) {
			auto pBox = (*it).predict();
			if (pBox.x >= 0 && pBox.y >= 0) {
				predictedBoxes.emplace_back(pBox);
				it++;
			}
			else {
				it = trackers.erase(it);
			}
		}
        
        // 3.2. associate detections to tracked object (both represented as bounding boxes)
		// dets : detFrameData[fi]
		auto trkNum = predictedBoxes.size();
		auto detNum = boxes.size();

		iouMatrix.clear();
		iouMatrix.resize(trkNum, vector<double>(detNum, 0));
        
		// compute iou matrix as a distance matrix
        for (unsigned int i = 0; i < trkNum; i++)  {
			for (unsigned int j = 0; j < detNum; j++) {
				// use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
				iouMatrix[i][j] = 1 - getIOU(
                    predictedBoxes[i], 
                    cv::Rect_<float>(
                        boxes[j].x, 
                        boxes[j].y, 
                        boxes[j].width, 
                        boxes[j].height));
			}
		}

        // solve the assignment problem using hungarian algorithm.
		// the resulting assignment is [track(prediction) : detection], with len=preNum
		HungarianAlgorithm HungAlgo;
		assignment.clear();
		HungAlgo.Solve(iouMatrix, assignment);

		// find matches, unmatched_detections and unmatched_predictions
		unmatchedTrajectories.clear();
		unmatchedDetections.clear();
		allItems.clear();
		matchedItems.clear();

		// there are unmatched detections
        if (detNum > trkNum) {
			for (unsigned int n = 0; n < detNum; n++)
				allItems.insert(n);

			for (unsigned int i = 0; i < trkNum; ++i)
				matchedItems.insert(assignment[i]);

			set_difference(allItems.begin(), allItems.end(),
				matchedItems.begin(), matchedItems.end(),
				insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
		}
		// there are unmatched trajectory/predictions
		else if (detNum < trkNum) {
			for (unsigned int i = 0; i < trkNum; ++i)
				if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
					unmatchedTrajectories.insert(i);
		}
		else {

		}
        
        // filter out matched with low IOU
		matchedPairs.clear();
		for (unsigned int i = 0; i < trkNum; ++i) {
			if (assignment[i] == -1) // pass over invalid values
				continue;
			if (1 - iouMatrix[i][assignment[i]] < _cfg.iou_thresh) {
				unmatchedTrajectories.insert(i);
				unmatchedDetections.insert(assignment[i]);
			}
			else {
				matchedPairs.emplace_back(cv::Point(i, assignment[i]));
			}
		}

        // 3.3. updating trackers
		// update matched trackers with assigned detections.
		// each prediction is corresponding to a tracker
		int detIdx, trkIdx;
		for (unsigned int i = 0; i < matchedPairs.size(); i++) {
			trkIdx = matchedPairs[i].x;
			detIdx = matchedPairs[i].y;
			trackers[trkIdx].update(
                cv::Rect_<float>(
                    boxes[detIdx].x, 
                    boxes[detIdx].y, 
                    boxes[detIdx].width, 
                    boxes[detIdx].height));
		}

		// create and initialise new trackers for unmatched detections
		for (auto& umd : unmatchedDetections) {
			auto tracker = KalmanTracker(
                cv::Rect_<float>(
                    boxes[umd].x, 
                    boxes[umd].y,
                    boxes[umd].width, 
                    boxes[umd].height));
			trackers.emplace_back(tracker);
		}

        // get trackers' output
		frameTrackingResult.clear();
		for (auto it = trackers.begin(); it != trackers.end();) {
			if (((*it).m_time_since_update < 1) &&
				((*it).m_hit_streak >= _cfg.min_hits)) {
				TrackingBox res;
				res.box = (*it).get_state();
				res.id = (*it).m_id + 1;
				frameTrackingResult.emplace_back(res);
				it++;
			}
			else
				it++;

			// remove dead tracker
			if (it != trackers.end() && (*it).m_time_since_update > _cfg.max_miss)
				it = trackers.erase(it);
		}

        for (const auto& tb : frameTrackingResult) {
			// id and box need to correspond
			for (int i = 0; i < boxes.size(); ++i) {
				if(getIOU(
                    cv::Rect_<float>(
                        boxes[i].x, 
                        boxes[i].y, 
                        boxes[i].width, 
                        boxes[i].height),
					cv::Rect_<float>(
                        tb.box.x, 
                        tb.box.y, 
                        tb.box.width, 
                        tb.box.height)) > 0.8) {
				    track_ids[i] = tb.id;
				}
			}
        }
        return;
    }
}