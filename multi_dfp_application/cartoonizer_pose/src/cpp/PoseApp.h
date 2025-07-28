#ifndef POSE_APP_H
#define POSE_APP_H

#include <atomic>
#include <mutex>
#include <deque>
#include <queue>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include "memx/accl/MxAccl.h"
#include <memx/mxutils/gui_view.h>

class PoseApp {
public:
    PoseApp(MX::Runtime::MxAccl* accl,
            const std::string& postproc_model_path,
            std::queue<cv::Mat>* frame_queue,
            std::mutex* queue_mutex,
            std::condition_variable* queue_cv,
            std::atomic_bool* runflag,
            MxQt* gui, int gui_stream_idx);

private:
    // Callback functions
    bool incallback_getframe(std::vector<const MX::Types::FeatureMap*> dst, int streamLabel);
    bool outcallback_getmxaoutput(std::vector<const MX::Types::FeatureMap*> src, int streamLabel);

    struct Box {
        float x1, y1, x2, y2, confidence;
        std::vector<std::pair<float, float>> keypoints;
    };

    // Runtime
    std::atomic_bool* runflag;
    std::queue<cv::Mat>* frame_queue;
    std::mutex* queue_mutex;
    std::condition_variable* queue_cv;
    std::deque<cv::Mat> frames_queue;
    std::mutex frameQueue_mutex;

    // GUI
    MxQt* gui_;
    int gui_stream_idx_;
    cv::Size displaySize;

    // Model
    std::string postproc_model_path;
    MX::Types::MxModelInfo model_info;
    MX::Types::MxModelInfo post_model_info;
    std::vector<float*> ofmap;

    // Preprocessing
    int model_input_width;
    int model_input_height;
    cv::Mat img_resized;
    cv::Mat img_model_in;

    // Pose postprocessing config
    float box_score;
    float kpt_score;
    float nms_thr;
    int dets_length;
    int num_kpts;

    const std::vector<cv::Scalar> COLOR_LIST;
    const std::vector<std::pair<int, int>> KEYPOINT_PAIRS;
};

#endif // POSE_APP_H
