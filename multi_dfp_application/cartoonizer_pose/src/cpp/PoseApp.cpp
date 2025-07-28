#include "PoseApp.h"
#include <opencv2/dnn/dnn.hpp>
#include <chrono>
#include <iostream>

using namespace std;

PoseApp::PoseApp(MX::Runtime::MxAccl* accl,
                 const std::string& postproc_model_path,
                 std::queue<cv::Mat>* frame_queue,
                 std::mutex* queue_mutex,
                 std::condition_variable* queue_cv,
                 std::atomic_bool* runflag,
                 MxQt* gui,
                 int gui_stream_idx)
    : frame_queue(frame_queue),
      queue_mutex(queue_mutex),
      queue_cv(queue_cv),
      runflag(runflag),
      postproc_model_path(postproc_model_path),
      box_score(0.25f), kpt_score(0.5f), nms_thr(0.2f), dets_length(8400), num_kpts(17),
      model_input_width(640), model_input_height(640),
      COLOR_LIST({
          {128,255,0}, {255,128,50}, {128,0,255}, {255,255,0}, {255,102,255},
          {255,51,255}, {51,153,255}, {255,153,153}, {255,51,51}, {153,255,153},
          {51,255,51}, {0,255,0}, {255,0,51}, {153,0,153}, {51,0,51},
          {0,0,0}, {0,102,255}, {0,51,255}, {0,153,255}, {0,153,153}
      }),
      KEYPOINT_PAIRS({
          {0, 1}, {0, 2}, {1, 3}, {2, 4}, {0, 5}, {0, 6}, {5, 7}, {7, 9},
          {6, 8}, {8, 10}, {5, 6}, {5, 11}, {6, 12}, {11, 12}, {11, 13},
          {13, 15}, {12, 14}, {14, 16}
      }),
      gui_(gui), gui_stream_idx_(gui_stream_idx)
{
    accl->connect_post_model(postproc_model_path);
    post_model_info = accl->get_post_model_info(0);
    model_info = accl->get_model_info(0);
    model_input_height = model_info.in_featuremap_shapes[0][0];
    model_input_width  = model_info.in_featuremap_shapes[0][1];

    ofmap.resize(post_model_info.num_out_featuremaps);
    for (int i = 0; i < post_model_info.num_out_featuremaps; ++i)
        ofmap[i] = new float[post_model_info.out_featuremap_sizes[i]];

    img_resized.create(model_input_height, model_input_width, CV_8UC3);
    img_model_in.create(model_input_height, model_input_width, CV_32FC3);

    displaySize = cv::Size(model_input_width, model_input_height);

    auto in_cb = std::bind(&PoseApp::incallback_getframe, this, std::placeholders::_1, std::placeholders::_2);
    auto out_cb = std::bind(&PoseApp::outcallback_getmxaoutput, this, std::placeholders::_1, std::placeholders::_2);
    accl->connect_stream(in_cb, out_cb, 21, 0);
}

bool PoseApp::incallback_getframe(std::vector<const MX::Types::FeatureMap*> dst, int streamLabel) {
    if (!runflag->load()) return false;

    cv::Mat inframe;
    {
        std::unique_lock<std::mutex> lock(*queue_mutex);
        queue_cv->wait(lock, [&]() { return !frame_queue->empty() || !runflag->load(); });

        if (!runflag->load()) return false;
        inframe = frame_queue->front().clone();
        frame_queue->pop();
    }

    {
        std::lock_guard<std::mutex> qlock(frameQueue_mutex);
        frames_queue.push_back(inframe.clone());
    }

    cv::Mat rgb, resized, floatImg;
    cv::cvtColor(inframe, rgb, cv::COLOR_BGR2RGB);
    cv::dnn::blobFromImage(rgb, resized, 1.0, cv::Size(model_input_width, model_input_height), cv::Scalar(0, 0, 0), true, false);
    resized.convertTo(floatImg, CV_32F, 1.0 / 255.0);
    dst[0]->set_data((float*)floatImg.data);

    return true;
}

bool PoseApp::outcallback_getmxaoutput(std::vector<const MX::Types::FeatureMap*> src, int streamLabel) {
    for (int i = 0; i < post_model_info.num_out_featuremaps; ++i)
        src[i]->get_data(ofmap[i]);

    cv::Mat inframe;
    {
        std::lock_guard<std::mutex> qlock(frameQueue_mutex);
        inframe = frames_queue.front().clone();
        frames_queue.pop_front();
    }

    float x_factor = inframe.cols / float(model_input_width);
    float y_factor = inframe.rows / float(model_input_height);

    std::vector<Box> boxes;
    std::vector<cv::Rect> rects;
    std::vector<float> scores;

    for (int i = 0; i < dets_length; ++i) {
        float conf = ofmap[0][4 * dets_length + i];
        if (conf < box_score) continue;

        float x = ofmap[0][0 * dets_length + i] * x_factor;
        float y = ofmap[0][1 * dets_length + i] * y_factor;
        float w = ofmap[0][2 * dets_length + i] * x_factor;
        float h = ofmap[0][3 * dets_length + i] * y_factor;

        int x1 = int(x - w / 2);
        int y1 = int(y - h / 2);
        int x2 = int(x + w / 2);
        int y2 = int(y + h / 2);

        Box box;
        box.confidence = conf;

        for (int j = 0; j < num_kpts; ++j) {
            float kx = ofmap[0][(5 + 3 * j + 0) * dets_length + i] * x_factor;
            float ky = ofmap[0][(5 + 3 * j + 1) * dets_length + i] * y_factor;
            float kc = ofmap[0][(5 + 3 * j + 2) * dets_length + i];
            box.keypoints.emplace_back(kc > kpt_score ? std::make_pair(kx, ky) : std::make_pair(-1.f, -1.f));
        }

        boxes.push_back(box);
        rects.emplace_back(x1, y1, x2 - x1, y2 - y1);
        scores.push_back(conf);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(rects, scores, box_score, nms_thr, keep);

    for (int idx : keep) {
        const auto& box = boxes[idx];
        for (const auto& pair : KEYPOINT_PAIRS) {
            auto p1 = box.keypoints[pair.first];
            auto p2 = box.keypoints[pair.second];
            if (p1.first != -1 && p2.first != -1)
                cv::line(inframe, {int(p1.first), int(p1.second)}, {int(p2.first), int(p2.second)}, {255,255,255}, 2);
        }

        for (int i = 0; i < box.keypoints.size(); ++i) {
            auto& kpt = box.keypoints[i];
            if (kpt.first != -1)
                cv::circle(inframe, {int(kpt.first), int(kpt.second)}, 3, COLOR_LIST[i % COLOR_LIST.size()], -1);
        }
    }

    cv::Mat bgrFrame;
    cv::cvtColor(inframe, bgrFrame, cv::COLOR_RGB2BGR);
    gui_->screens[0]->SetDisplayFrame(gui_stream_idx_, &bgrFrame);  

    return true;
}

