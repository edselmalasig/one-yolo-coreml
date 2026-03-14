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
const std::vector<std::string> VIDEO_SOURCES = {
    "assets/cam1-480p.mov", "assets/cam2-480p.mov", "assets/cam3-480p.mov",
    "assets/cam4-480p.mov", "assets/cam5-480p.mov", "assets/cam6-480p.mov",
    "assets/cam1-480p.mov", "assets/cam2-480p.mov"
};

// ---------------------------------------------------------------------------
// Rounded-corner label helper
// ---------------------------------------------------------------------------
static void drawRoundedLabel(
    cv::Mat&           img,
    const std::string& text,
    cv::Point          origin,       // bottom-left of text
    int                font       = cv::FONT_HERSHEY_SIMPLEX,
    double             font_scale = 0.75,
    int                thickness  = 2,
    cv::Scalar         text_color = cv::Scalar(255, 255, 255),
    cv::Scalar         bg_color   = cv::Scalar(0, 0, 0),
    int                radius     = 10,
    int                pad_x      = 10,
    int                pad_y      = 6,
    double             bg_alpha   = 0.65)
{
    int baseline = 0;
    cv::Size ts = cv::getTextSize(text, font, font_scale, thickness, &baseline);

    cv::Point tl(origin.x - pad_x,
                 origin.y - ts.height - pad_y);
    cv::Point br(origin.x + ts.width  + pad_x,
                 origin.y + baseline  + pad_y);

    tl.x = std::max(tl.x, 0);
    tl.y = std::max(tl.y, 0);
    br.x = std::min(br.x, img.cols - 1);
    br.y = std::min(br.y, img.rows - 1);

    cv::Rect badge_rect(tl, br);
    if (badge_rect.width <= 0 || badge_rect.height <= 0) return;

    cv::Mat overlay = img.clone();
    int r = std::min(radius, std::min(badge_rect.width, badge_rect.height) / 2);

    cv::rectangle(overlay, badge_rect, bg_color, cv::FILLED);
    cv::rectangle(overlay,
                  cv::Point(tl.x + r, tl.y),
                  cv::Point(br.x - r, br.y),
                  bg_color, cv::FILLED);
    cv::rectangle(overlay,
                  cv::Point(tl.x, tl.y + r),
                  cv::Point(br.x, br.y - r),
                  bg_color, cv::FILLED);
    auto drawCorner = [&](cv::Point center) {
        cv::circle(overlay, center, r, bg_color, cv::FILLED, cv::LINE_AA);
    };
    drawCorner(cv::Point(tl.x + r, tl.y + r));
    drawCorner(cv::Point(br.x - r, tl.y + r));
    drawCorner(cv::Point(tl.x + r, br.y - r));
    drawCorner(cv::Point(br.x - r, br.y - r));

    cv::addWeighted(overlay, bg_alpha, img, 1.0 - bg_alpha, 0, img);

    cv::putText(img, text, origin,
                font, font_scale, text_color, thickness, cv::LINE_AA);
}

// ---------------------------------------------------------------------------
// Thread-safe Frame Queue
// ---------------------------------------------------------------------------
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
    const std::chrono::duration<double> frame_duration(1.0 / 60.0);

    while (!stopped) {
        auto frame_start = std::chrono::steady_clock::now();

        cv::Mat frame;
        if (!cap.read(frame)) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }
        fq.push(frame);

        auto elapsed    = std::chrono::steady_clock::now() - frame_start;
        auto sleep_time = frame_duration - elapsed;
        if (sleep_time > std::chrono::duration<double>(0))
            std::this_thread::sleep_for(sleep_time);
    }
}

// --- INFERENCE THREAD ---
void inferenceWorker(FrameQueue& fq, InferenceState& state, int id) {
    YoloConfig cfg;
    cfg.model_path  = "models/yolo26s.mlpackage";
    cfg.task        = YoloTaskType::DET;
    cfg.version     = YoloVersion::YOLO26;
    cfg.target_rt   = YoloTargetRT::CML;
    cfg.conf_thresh = 0.4f;

    yolo::Yolo model(cfg);

    auto prev_time = std::chrono::steady_clock::now();

    while (!state.stopped) {
        cv::Mat frame;
        if (!fq.pop(frame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        YoloResult result = model.predict(frame);
        cv::Mat annotated = result.plot();

        auto curr_time = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(curr_time - prev_time).count();
        if (dt <= 0.0f) dt = 1e-6f;
        prev_time = curr_time;

        {
            std::lock_guard<std::mutex> lock(state.mtx);
            state.fps = (0.9f * state.fps) + (0.1f * (1.0f / dt));

            drawRoundedLabel(
                annotated,
                "CH" + std::to_string(id + 1) + "  FPS: " + std::to_string((int)state.fps),
                cv::Point(30, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.75, 2,
                cv::Scalar(255, 255, 255),
                cv::Scalar(10,  10,  10),
                12, 12, 8, 0.75
            );

            state.current_frame = annotated.clone();
        }
    }
}


int main() {
    const int NUM_CAMS = static_cast<int>(VIDEO_SOURCES.size()); // 8

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

    // View modes:
    //   0      = 4x2 grid
    //   1–8    = individual channel fullscreen
    //   9      = 'f' mode: 4x2 grid with source filename labels
    int current_view = 0;

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

        if (current_view == 9) {
            // --- 'f' mode: 4x2 grid with rounded source-filename banners ---
            std::vector<cv::Mat> resized(NUM_CAMS);
            for (int i = 0; i < NUM_CAMS; ++i) {
                cv::resize(display_frames[i], resized[i], cv::Size(640, 360));

                // Source filename — bottom
                drawRoundedLabel(
                    resized[i],
                    VIDEO_SOURCES[i],
                    cv::Point(14, 340),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65, 2,
                    cv::Scalar(255, 255, 255),
                    cv::Scalar(10,  10,  10),
                    10, 10, 7, 0.60
                );

                // SRC badge — top-left
                drawRoundedLabel(
                    resized[i],
                    "SRC " + std::to_string(i + 1),
                    cv::Point(14, 38),
                    cv::FONT_HERSHEY_SIMPLEX, 0.75, 2,
                    cv::Scalar(0, 220, 255),
                    cv::Scalar(10,  10,  10),
                    10, 10, 7, 0.60
                );
            }

            cv::Mat row0, row1;
            cv::hconcat(std::vector<cv::Mat>{resized[0], resized[1], resized[2], resized[3]}, row0);
            cv::hconcat(std::vector<cv::Mat>{resized[4], resized[5], resized[6], resized[7]}, row1);
            cv::vconcat(row0, row1, canvas);

            // Title bar
            cv::Mat title_bar(50, canvas.cols, CV_8UC3, cv::Scalar(20, 20, 20));
            drawRoundedLabel(
                title_bar,
                "ALL VIDEO SOURCES  [f: toggle | 0: grid | 1-8: channel | q: quit]",
                cv::Point(16, 38),
                cv::FONT_HERSHEY_SIMPLEX, 0.85, 2,
                cv::Scalar(0, 220, 255),
                cv::Scalar(20, 20, 20),
                0, 0, 0, 0.0
            );
            cv::vconcat(title_bar, canvas, canvas);

        } else if (current_view == 0) {
            // Standard 4x2 grid
            std::vector<cv::Mat> resized(NUM_CAMS);
            for (int i = 0; i < NUM_CAMS; ++i)
                cv::resize(display_frames[i], resized[i], cv::Size(640, 360));

            cv::Mat row0, row1;
            cv::hconcat(std::vector<cv::Mat>{resized[0], resized[1], resized[2], resized[3]}, row0);
            cv::hconcat(std::vector<cv::Mat>{resized[4], resized[5], resized[6], resized[7]}, row1);
            cv::vconcat(row0, row1, canvas);

        } else {
            // Individual channel fullscreen (1–8)
            canvas = display_frames[current_view - 1];
        }

        cv::imshow("8-Way Drive Assist", canvas);

        char key = (char)cv::waitKey(1);
        if (key == 'q') break;
        if (key == 'f') current_view = (current_view == 9) ? 0 : 9;
        if (key >= '1' && key <= '8') current_view = key - '0';
        if (key == '0') current_view = 0;
    }

    global_stop = true;
    for (auto s : states) s->stopped = true;
    for (auto& t : threads) if (t.joinable()) t.join();

    for (auto q : queues) delete q;
    for (auto s : states) delete s;

    return 0;
}