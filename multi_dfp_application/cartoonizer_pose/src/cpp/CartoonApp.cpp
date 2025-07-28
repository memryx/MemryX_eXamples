#include "CartoonApp.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>

using namespace std;

CartoonApp::CartoonApp(MX::Runtime::MxAccl* accl,
                       std::queue<cv::Mat>* frame_queue,
                       std::mutex* queue_mutex,
                       std::condition_variable* queue_cv,
                       std::atomic_bool* runflag,
                       MxQt* gui,
                       int gui_stream_idx)
    : frame_queue(frame_queue), queue_mutex(queue_mutex), queue_cv(queue_cv),
      runflag(runflag), gui_(gui), gui_stream_idx_(gui_stream_idx)
{
    model_info = accl->get_model_info(0);
    model_input_height = model_info.in_featuremap_shapes[0][0];
    model_input_width = model_info.in_featuremap_shapes[0][1];
    model_output_height = model_info.out_featuremap_shapes[0][0];
    model_output_width = model_info.out_featuremap_shapes[0][1];

    img_resized.create(model_input_height, model_input_width, CV_8UC3);
    img_model_in.create(model_input_height, model_input_width, CV_32FC3);
    img_model_out.create(cv::Size(model_output_height, model_output_width), CV_32FC3);
    img_model_out_uint.create(img_model_out.size(), CV_8UC3);

    displaySize = cv::Size(model_input_width, model_input_height);
    img_final_out_resized.create(displaySize, CV_8UC3);

    // Bind input/output callbacks
    auto in_cb = std::bind(&CartoonApp::incallback_getframe, this, std::placeholders::_1, std::placeholders::_2);
    auto out_cb = std::bind(&CartoonApp::outcallback_getmxaoutput, this, std::placeholders::_1, std::placeholders::_2);
    accl->connect_stream(in_cb, out_cb, 20, 0);
}

bool CartoonApp::incallback_getframe(std::vector<const MX::Types::FeatureMap*> dst, int streamLabel)
{
    if (!runflag->load()) return false;

    cv::Mat inframe;
    {
        std::unique_lock<std::mutex> lock(*queue_mutex);
        queue_cv->wait(lock, [&]() { return !frame_queue->empty() || !runflag->load(); });

        if (!runflag->load()) return false;
        inframe = frame_queue->front().clone();
        frame_queue->pop();
    }

    cv::resize(inframe, img_resized, img_resized.size());
    cv::cvtColor(img_resized, img_resized, cv::COLOR_BGR2RGB);
    img_resized.convertTo(img_model_in, CV_32FC3, 1.0 / 127.5, -1.0);
    dst[0]->set_data((float*)img_model_in.data);
    
    return true;
}

bool CartoonApp::outcallback_getmxaoutput(std::vector<const MX::Types::FeatureMap*> src, int streamLabel)
{
    src[0]->get_data((float*)img_model_out.data);

    // Convert [-1,1] → [0,255] safely
    cv::add(img_model_out, cv::Scalar(1.0f, 1.0f, 1.0f), img_model_out);       
    cv::min(cv::max(img_model_out, 0.0f), 2.0f, img_model_out);       
    img_model_out.convertTo(img_model_out_uint, CV_8UC3, 127.5);                 

    // Resize for display and show
    cv::resize(img_model_out_uint, img_final_out_resized, displaySize);
    gui_->screens[0]->SetDisplayFrame(gui_stream_idx_, &img_final_out_resized);  

    return true;
}
