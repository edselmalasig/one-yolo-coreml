#include "Yolo.h"
#include "track/YoloTracker.h"
using namespace yolo;

int main() {
    /* 1. construct YoloConfig */
    YoloConfig cfg;
    cfg.desc        = "vehicle keypoint estimation task(2 points used for location) using yolov5su(anchor-free, custom model), using default(OpenCV::DNN/CUDA) runtime.";
    cfg.version     = YoloVersion::YOLO5U;
    cfg.task        = YoloTaskType::POSE;
    cfg.target_rt   = YoloTargetRT::OPENCV_CUDA;
    cfg.model_path  = "./vp_data/models/det_cls/vehicle_keypoint_v5su_c5_20260310.onnx";
    cfg.input_w     = 640;
    cfg.input_h     = 384;
    cfg.batch_size  = 1;
    cfg.num_classes = 5;
    cfg.num_kpts    = 2;     // only 2 keypoints for custom model
    cfg.names       = {"person", "car", "bus", "truck", "2wheel"};

    /* 2. create Yolo using YoloConfig */
    auto model = Yolo(cfg);
    model.info();

    /* 3. construct YoloTrackConfig */
    YoloTrackConfig t_cfg;
    t_cfg.algo = YoloTrackAlgo::SORT;

    /* 4. create YoloTracker using YoloTrackConfig */
    auto tracker = YoloTracker(t_cfg);
    tracker.info();

    /* 5. open video and predict frames in a loop */
    cv::VideoCapture cap("./vp_data/test_video/smoke2.mp4");
    while (cap.isOpened()) {
        // collect frame
        cv::Mat frame;
        if (!cap.read(frame)) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        // predict with batch mode (batch size == 1)
        auto results = model(std::vector<cv::Mat>{frame});
        
        // track result
        tracker(results[0]);

        // show and print
        results[0].info();           // print summary
        results[0].to_json(true);    // convert structured result to json and print
        results[0].to_csv(true);     // convert structured result to csv and print

        DrawParam draw_param;        // redefine DrawParam parameter because we used a custom pose estimation model， which output only 2 keypoints 
        draw_param.kpt_pairs       = {{0, 1}};                    // link the 1st and 2nd poits
        draw_param.kpt_pair_colors = {cv::Scalar(0, 255, 255)};   // color to the link line
        if (results[0].show(
            false, 0.7f, draw_param,  // show annotated image & input image(640*384) with unblock mode
            false, true) == 27) {     // exit loop if user has pressed ESC
            break;
        }

        /*
         * you can also get structured results like below:
         * auto boxes   = results[0].boxes();               // get bounding boxes in pose task
         * auto cls_ids = results[0].cls_ids();             // get class ids in pose task
         * auto confs   = results[0].confs();               // get confidences in pose task
         * auto labels  = results[0].labels();              // get labels in pose task
         * auto kpts    = results[0].kpts();                // get keypoints in pose task
         * auto track_ids    = results[0].track_ids();      // get track ids in pose task
         * auto track_points = results[0].track_points();   // get track points in pose task
        */
    }
}