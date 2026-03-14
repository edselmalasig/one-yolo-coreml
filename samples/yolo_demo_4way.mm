#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <opencv2/opencv.hpp>
#include "Yolo.h" // from one-yolo

using namespace yolo;

// Configuration
const std::vector<std::string> VIDEO_SOURCES = {"assets/cam1-480p.mov", "assets/cam2-480p.mov", "assets/cam3-480p.mov", "assets/cam4-480p.mov"};

// Thread-safe Frame Queue
class FrameQueue {
    std::queue<cv::Mat> queue;
    std::mutex mtx;
    const size_t max_size = 32;
public:
    void push(cv::Mat frame) {
        std::lock_guard<std::mutex> lock(mtx);
        if (queue.size() < max_size) queue.push(frame);
    }
    bool pop(cv::Mat& frame) {
        std::lock_guard<std::mutex> lock(mtx);
        if (queue.empty()) return false;
        frame = queue.front();
        queue.pop();
        return true;
    }
};

struct InferenceState {
    cv::Mat current_frame;
    float fps = 0;
    std::mutex mtx;
    std::atomic<bool> stopped{false};
};

// --- LOADER THREAD ---
void videoLoader(std::string source, FrameQueue& fq, std::atomic<bool>& stopped) {
    cv::VideoCapture cap(source);
    while (!stopped) {
        cv::Mat frame;
        if (!cap.read(frame)) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }
        fq.push(frame);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

// --- INFERENCE THREAD ---
void inferenceWorker(FrameQueue& fq, InferenceState& state, int id) {
    // Initialize one-yolo with CoreML backend (Apple Silicon — ANE/GPU/CPU)
    YoloConfig cfg;
    cfg.model_path = "models/yolo26n.mlpackage";
    cfg.task       = YoloTaskType::DET;
    cfg.version    = YoloVersion::YOLO26;
    cfg.target_rt  = YoloTargetRT::CML;
    cfg.conf_thresh = 0.4f;

    yolo::Yolo model(cfg);

    auto prev_time = std::chrono::steady_clock::now();

    while (!state.stopped) {
        cv::Mat frame;
        if (!fq.pop(frame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Run Inference
        YoloResult result = model.predict(frame);

        // Annotate using the result's plot() method
        cv::Mat annotated = result.plot();

        auto curr_time = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(curr_time - prev_time).count();
        if (dt <= 0.0f) dt = 1e-6f; // guard against division by zero
        prev_time = curr_time;

        {
            std::lock_guard<std::mutex> lock(state.mtx);
            state.fps = (0.9f * state.fps) + (0.1f * (1.0f / dt));
            cv::putText(annotated, "CH" + std::to_string(id + 1) + " | FPS: " + std::to_string((int)state.fps),
                        cv::Point(30, 60), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 0), 3);
            state.current_frame = annotated.clone();
        }
    }
}

int main() {
    const int NUM_CAMS = static_cast<int>(VIDEO_SOURCES.size());

    std::vector<FrameQueue*>     queues;
    std::vector<InferenceState*> states;
    std::vector<std::thread>     threads;
    std::atomic<bool> global_stop{false};

    for (int i = 0; i < NUM_CAMS; ++i) {
        queues.push_back(new FrameQueue());
        states.push_back(new InferenceState());
        threads.emplace_back(videoLoader, VIDEO_SOURCES[i], std::ref(*queues[i]), std::ref(global_stop));
        threads.emplace_back(inferenceWorker, std::ref(*queues[i]), std::ref(*states[i]), i);
    }

    int current_view = 0; // 0: Grid, 1-4: Individual
    while (true) {
        std::vector<cv::Mat> display_frames(NUM_CAMS);
        bool all_ready = true;

        for (int i = 0; i < NUM_CAMS; ++i) {
            std::lock_guard<std::mutex> lock(states[i]->mtx);
            if (states[i]->current_frame.empty()) { all_ready = false; break; }
            display_frames[i] = states[i]->current_frame.clone();
        }

        if (!all_ready) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        cv::Mat canvas;
        if (current_view == 0) {
            // Resize into temporaries to avoid aliasing when resizing in-place
            std::vector<cv::Mat> resized(NUM_CAMS);
            for (int i = 0; i < NUM_CAMS; ++i)
                cv::resize(display_frames[i], resized[i], cv::Size(640, 480));
            cv::Mat top, bot;
            cv::hconcat(resized[0], resized[1], top);
            cv::hconcat(resized[2], resized[3], bot);
            cv::vconcat(top, bot, canvas);
        } else {
            canvas = display_frames[current_view - 1];
        }

        cv::imshow("4-Way Monitor", canvas);
        char key = (char)cv::waitKey(1);
        if (key == 'q') break;
        if (key >= '1' && key <= '4') current_view = key - '0';
        if (key == '0') current_view = 0;
    }

    global_stop = true;
    for (auto s : states) s->stopped = true;
    for (auto& t : threads) if (t.joinable()) t.join();

    for (auto q : queues) delete q;
    for (auto s : states) delete s;

    return 0;
}