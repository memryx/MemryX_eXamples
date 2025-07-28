#ifndef CARTOON_APP_H
#define CARTOON_APP_H

#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include "memx/accl/MxAccl.h"
#include <memx/mxutils/gui_view.h>

class CartoonApp {
public:
    CartoonApp(MX::Runtime::MxAccl* accl,
               std::queue<cv::Mat>* frame_queue,
               std::mutex* queue_mutex,
               std::condition_variable* queue_cv,
               std::atomic_bool* runflag,
               MxQt* gui,
               int gui_stream_idx);

private:
    // Input/output callbacks
    bool incallback_getframe(std::vector<const MX::Types::FeatureMap*> dst, int streamLabel);
    bool outcallback_getmxaoutput(std::vector<const MX::Types::FeatureMap*> src, int streamLabel);

    // Runtime state
    std::atomic_bool* runflag;
    std::queue<cv::Mat>* frame_queue;
    std::mutex* queue_mutex;
    std::condition_variable* queue_cv;

    // GUI
    MxQt* gui_;
    int gui_stream_idx_;
    cv::Size displaySize;

    // Model info
    MX::Types::MxModelInfo model_info;
    int model_input_height;
    int model_input_width;
    int model_output_height;
    int model_output_width;

    // Image buffers
    cv::Mat img_resized;
    cv::Mat img_model_in;
    cv::Mat img_model_out;
    cv::Mat img_model_out_uint;
    cv::Mat img_final_out_resized;
};

#endif // CARTOON_APP_H
