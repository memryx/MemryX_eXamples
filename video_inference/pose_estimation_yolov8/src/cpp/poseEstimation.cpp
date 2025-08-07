#include <iostream>
#include <thread>
#include <signal.h>
#include <atomic>
#include <deque>
#include <opencv2/opencv.hpp>    /* imshow */
#include <opencv2/imgproc.hpp>   /* cvtcolor */
#include <opencv2/imgcodecs.hpp> /* imwrite */
#include <filesystem>
#include "memx/accl/MxAccl.h"

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace fs = std::filesystem;

bool use_cam = true; // Flag to determine whether to use the camera or video
std::atomic_bool runflag; // Atomic flag to control the processing loop
bool window_created = false; // Flag to ensure window is only created once

// Model file paths
const fs::path modelPath = "YOLO_v8_medium_pose_640_640_3_onnx.dfp"; // Path to DFP model
const fs::path onnx_postprocessing_model_path = "YOLO_v8_medium_pose_640_640_3_onnx_post.onnx"; // Path to post-processing ONNX model
std::string server_addr = "/run/mxa_manager/";
fs::path videoPath;

// Model information containers
MX::Types::MxModelInfo model_info; // Information for the DFP model
MX::Types::MxModelInfo post_model_info; // Information for post-processing model

std::vector<float*> ofmap; // Container for output feature maps

// Model input parameters
int model_input_width = 640;
int model_input_height = 640;
double origHeight = 0.0;  // Original frame height
double origWidth = 0.0;  // Original frame width

float box_score = 0.25; // Threshold for box confidence
float rat = 0.0; // Aspect ratio used during resizing
float kpt_score = 0.5; // Threshold for keypoint confidence
float nms_thr = 0.2; // IoU threshold for Non-Maximum Suppression (NMS)
int dets_length = 8400; // Number of detections
int num_kpts = 17; // Number of keypoints in pose estimation

// OpenCV video capture object
cv::VideoCapture vcap;

// Queue to store frames for processing
std::deque<cv::Mat> frames_queue;
const int max_backlog = 5;
std::mutex frameQueue_mutex; // Mutex for frame queue access

#define AVG_FPS_CALC_FRAME_COUNT  50 // Number of frames to calculate FPS over
int frame_count = 0;
float fps_number = .0;
char fps_text[64] = "FPS = ";
std::chrono::milliseconds start_ms; // Variable to store start time for FPS calculation

// Color list for drawing keypoints
const std::vector<cv::Scalar> COLOR_LIST = {
    cv::Scalar(128, 255, 0), cv::Scalar(255, 128, 50), cv::Scalar(128, 0, 255),
    cv::Scalar(255, 255, 0), cv::Scalar(255, 102, 255), cv::Scalar(255, 51, 255),
    cv::Scalar(51, 153, 255), cv::Scalar(255, 153, 153), cv::Scalar(255, 51, 51),
    cv::Scalar(153, 255, 153), cv::Scalar(51, 255, 51), cv::Scalar(0, 255, 0),
    cv::Scalar(255, 0, 51), cv::Scalar(153, 0, 153), cv::Scalar(51, 0, 51),
    cv::Scalar(0, 0, 0), cv::Scalar(0, 102, 255), cv::Scalar(0, 51, 255),
    cv::Scalar(0, 153, 255), cv::Scalar(0, 153, 153)
};

// Pairs of keypoints for drawing skeleton
const std::vector<std::pair<int, int>> KEYPOINT_PAIRS = {
    {0, 1}, {0, 2}, {1, 3}, {2, 4}, {0, 5}, {0, 6}, {5, 7}, {7, 9}, {6, 8},
    {8, 10}, {5, 6}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}
};

// Signal handler for cleanly exiting on interrupt (Ctrl + C)
void signalHandler(int pSignal) {
    runflag.store(false);
}

// Callback function for processing input frames
bool incallback_getframe(std::vector<const MX::Types::FeatureMap*> dst, int streamLabel) {

    if (runflag.load()) {
        cv::Mat inframe;


        while (true) {
            // Capture a frame from video/camera
            bool got_frame = vcap.read(inframe);

            if (!got_frame) {
                std::cout << "\n\n No frame - End of cam? \n\n\n";
                runflag.store(false);
                return false;
            }
            else {
                // Push the captured frame to the queue
                {
                    std::lock_guard<std::mutex> flock(frameQueue_mutex);
                    // only push this frame if there's space
                    if (use_cam && (frames_queue.size() >= max_backlog)) {
                        // drop it and capture the next frame instead
                        continue;
                    }
                    else {
                        frames_queue.push_back(inframe);
                    }
                }

                // Convert frame to RGB and preprocess for model input
                cv::Mat rgbImage;
                cv::cvtColor(inframe, rgbImage, cv::COLOR_BGR2RGB);
                cv::Mat preProcframe;
                // cv::resize(rgbImage, preProcframe, 1.0, cv::Size(model_input_width, model_input_height), cv::Scalar(0, 0, 0), true, false);
                cv::dnn::blobFromImage(rgbImage, preProcframe, 1.0, cv::Size(model_input_width, model_input_height), cv::Scalar(0, 0, 0), true, false);
                cv::Mat floatImage;
                preProcframe.convertTo(floatImage, CV_32F, 1.0 / 255.0); // Normalize the frame

                // Set preprocessed input data to accelerator
                dst[0]->set_data((float*)floatImage.data);
                return true;
            }
        }
    }
    else {
        vcap.release(); // Release video resources if runflag is false
        return false;
    }
}

// Box structure to hold bounding box and keypoints
struct Box {
    float x1, y1, x2, y2, confidence;
    std::vector<std::pair<float, float>> keypoints; // Keypoints (x, y)
};

// Output callback function to process model output
bool outcallback_getmxaoutput(std::vector<const MX::Types::FeatureMap*> src, int streamLabel) {

    // Get data from the feature maps
    for (int i = 0; i < post_model_info.num_out_featuremaps; ++i) {
        src[i]->get_data(ofmap[i]);
    }

    // Get the input frame from the queue
    cv::Mat inframe;
    {
        std::lock_guard<std::mutex> flock(frameQueue_mutex);
        inframe = frames_queue.front();
        frames_queue.pop_front();
    }

    // Create a window for displaying results if not created yet
    if (!window_created) {
        cv::namedWindow("Pose Estimation", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
        cv::resizeWindow("Pose Estimation", cv::Size(origWidth, origHeight));
        cv::moveWindow("Pose Estimation", 0, 0);
        window_created = true;
    }

    // Process detection results and perform NMS
    std::vector<Box> all_boxes;
    std::vector<float> all_scores;
    std::vector<cv::Rect> cv_boxes;

    // Loop through each detection
#pragma omp parallel for num_threads(2)
    for (int i = 0; i < dets_length; ++i) {

        // Extract bounding box coordinates and confidence score
        float x0 = ofmap[0][dets_length * 0 + i];
        float y0 = ofmap[0][dets_length * 1 + i];
        float w = ofmap[0][dets_length * 2 + i];
        float h = ofmap[0][dets_length * 3 + i];
        float confidence = ofmap[0][dets_length * 4 + i];

        if (confidence > box_score) {
            Box box;
            box.confidence = confidence;

            // Scale box coordinates back to original image size
            float y_factor = inframe.rows / float(model_input_height);
            float x_factor = inframe.cols / float(model_input_width);
            x0 = x0 * x_factor;
            y0 = y0 * y_factor;
            w = w * x_factor;
            h = h * y_factor;

            int x1 = (int)(x0 - 0.5 * w);
            int x2 = (int)(x0 + 0.5 * w);
            int y1 = (int)(y0 - 0.5 * h);
            int y2 = (int)(y0 + 0.5 * h);

            // Extract keypoints for pose estimation
            for (int j = 0; j < num_kpts; ++j) {
                float kpt_x = ofmap[0][dets_length * (5 + j * 3) + i] * x_factor;
                float kpt_y = ofmap[0][dets_length * (5 + j * 3 + 1) + i] * y_factor;
                float kpt_conf = ofmap[0][dets_length * (5 + j * 3 + 2) + i];

                // Add keypoints if confidence is above threshold
                if (kpt_conf > kpt_score) {
                    box.keypoints.push_back(std::make_pair(kpt_x, kpt_y));
                }
                else {
                    box.keypoints.push_back(std::make_pair(-1, -1)); // Invalid keypoint
                }
            }

#pragma omp critical
            {
                all_boxes.push_back(box);
                all_scores.push_back(confidence);
                cv_boxes.push_back(cv::Rect(x1, y1, x2 - x1, y2 - y1));
            }
        }
    }

    // Apply Non-Maximum Suppression (NMS) to reduce overlapping boxes
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(cv_boxes, all_scores, box_score, nms_thr, nms_result);

    // Keep only the filtered boxes after NMS
    std::vector<Box> filtered_boxes;
    for (int idx : nms_result) {
        filtered_boxes.push_back(all_boxes[idx]);
    }

    // Draw keypoints and connections (skeleton) on the frame
    for (const auto& box : filtered_boxes) {
        for (const auto& connection : KEYPOINT_PAIRS) {
            int idx1 = connection.first;
            int idx2 = connection.second;

            if (idx1 < box.keypoints.size() && idx2 < box.keypoints.size()) {
                auto kpt1 = box.keypoints[idx1];
                auto kpt2 = box.keypoints[idx2];

                if (kpt1.first != -1 && kpt1.second != -1 && kpt2.first != -1 && kpt2.second != -1) {
                    cv::line(inframe, cv::Point(kpt1.first, kpt1.second), cv::Point(kpt2.first, kpt2.second), cv::Scalar(255, 255, 255), 3);
                }
            }
        }

        // Draw individual keypoints
        for (int k = 0; k < box.keypoints.size(); ++k) {
            auto& kpt = box.keypoints[k];
            if (kpt.first != -1 && kpt.second != -1) {
                cv::circle(inframe, cv::Point(kpt.first, kpt.second), 4, COLOR_LIST[k % COLOR_LIST.size()], -1);
            }
        }
    }

    // Calculate FPS every AVG_FPS_CALC_FRAME_COUNT frames
    frame_count++;
    if (frame_count == 1) {
        start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    }
    else if (frame_count % AVG_FPS_CALC_FRAME_COUNT == 0) {
        std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()) - start_ms;
        fps_number = (float)AVG_FPS_CALC_FRAME_COUNT * 1000 / (float)(duration.count());
        sprintf(fps_text, "FPS = %.1f", fps_number);
        frame_count = 0;
    }

    // Display FPS on the frame
    cv::putText(inframe, fps_text, cv::Point2i(450, 30), cv::FONT_ITALIC, 0.8, cv::Scalar(255, 255, 255), 2);

    // Show the frame with keypoints and bounding boxes
    cv::imshow("Pose Estimation", inframe);

    // Exit if 'q' key is pressed
    if (cv::waitKey(1) == 'q') {
        runflag.store(false);
    }

    return true;
}

// Function to start inference
void run_inference() {

    runflag.store(true);

    if (use_cam) {
        std::cout << "use cam";
        vcap.open(0, cv::CAP_ANY); // Open camera
    }
    else {
        vcap.open(videoPath.string()); // Open video file
    }

    std::cout<<"check use cam\n";

    if (vcap.isOpened()) {
        std::cout << "videocapture opened \n";
        origWidth = vcap.get(cv::CAP_PROP_FRAME_WIDTH); // Get original frame width
        origHeight = vcap.get(cv::CAP_PROP_FRAME_HEIGHT); // Get original frame height
    }
    else {
        std::cout << "videocapture NOT opened \n";
        runflag.store(false);
    }



    if (runflag.load()) {

        #ifdef _WIN32
            // Initialize the MemryX accelerator
            server_addr = "localhost";
        #endif

        MX::Runtime::MxAccl accl{ 
                    fs::path(modelPath),                      // DFP path
                    std::vector<int>{0},                    // device_ids_to_use
                    std::array<bool, 2>{true, true},        // use_model_shape
                    false,                                  // local_mode
                    MX::RPC::SchedulerOptions{600, 0, false, 16, 12},  // sched_options
                    MX::RPC::ClientOptions{false, 0},       // client_options
                    server_addr,                            // server_addr
                    10000,                                  // server_port_base
                    false };

        accl.connect_post_model(fs::path(onnx_postprocessing_model_path)); // Connect the post-processing model
        post_model_info = accl.get_post_model_info(0); // Get post-processing model info

        model_info = accl.get_model_info(0); // Get main model info
        model_input_height = model_info.in_featuremap_shapes[0][0]; // Set model input height
        model_input_width = model_info.in_featuremap_shapes[0][1]; // Set model input width

        // Allocate memory for feature maps
        ofmap.reserve(post_model_info.num_out_featuremaps);
        for (int i = 0; i < post_model_info.num_out_featuremaps; ++i) {
            ofmap.push_back(new float[post_model_info.out_featuremap_sizes[i]]);
        }

        // Connect input and output streams to the accelerator
        accl.connect_stream(&incallback_getframe, &outcallback_getmxaoutput, 10 /*unique stream ID*/, 0 /*Model ID*/);

        std::cout << "Connected stream \n\n\n";
        accl.start(); // Start inference
        accl.wait();  // Wait for inference to complete
        accl.stop();  // Stop the accelerator

        // Clean up allocated memory
        for (auto& fmap : ofmap) {
            delete[] fmap;
            fmap = NULL;
        }
        std::cout << "\n\rAccl stop called \n";
    }
}

// Main function to handle command-line arguments and start the app
int main(int argc, char* argv[]) {

    if (argc > 1) {

        std::string inputType(argv[1]);

        if (inputType == "--cam") {
            use_cam = true;
            runflag.store(true); // Use camera
        }
        else if (inputType == "--video") {
            use_cam = false;
            if (argc > 2) {
                videoPath = std::filesystem::path(argv[2]);  // Get the video path from the command line
            }
            else {
                std::cout << "Error: Missing video path after --video\n";
                return 1;
            }
            runflag.store(true); // Use video
        }
        else {
            std::cout << "\n\nIncorrect Argument Passed \n\tuse ./app [--cam] or [--video <path-to-video>]\n\n\n";
            runflag.store(false);
        }

    }
    else {
        std::cout << "\n\nNo Arguments Passed \n\tuse ./app [--cam] or [--video <path-to-video>]\n\n\n";
        runflag.store(false);
    }

    // Handle signal (Ctrl+C) to gracefully stop the app
    signal(SIGINT, signalHandler);

    if (runflag.load()) {

        std::cout << "application start \n";
        std::cout << "model path = " << modelPath.string() << "\n";

        // Start the inference process
        run_inference();
    }

    else {
        std::cout << "App exiting without execution \n\n\n";
    }

    return 1;
}
