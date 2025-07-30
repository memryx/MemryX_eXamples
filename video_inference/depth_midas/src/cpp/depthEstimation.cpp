#include <iostream>
#include <signal.h>
#include <opencv2/opencv.hpp>    /* imshow */
#include <opencv2/imgproc.hpp>   /* cvtcolor */
#include <opencv2/imgcodecs.hpp> /* imwrite */
#include "memx/accl/MxAccl.h"
#include <filesystem>
#include <string>

namespace fs = std::filesystem;
std::string server_addr = "/run/mxa_manager/";

#define AVG_FPS_CALC_FRAME_COUNT  50

// Command-line argument flags
std::string modelPath;
fs::path videoPath;

atomic_bool runflag(true);
void signalHandler(int pSignal){
    runflag.store(false);
}

// In case of cameras try to use best possible input configurations which are setting the
// resolution to 640x480 and try to set the input FPS to 30
bool configureCamera(cv::VideoCapture& vcap) {
    bool settings_success = true;

    try {
        if (!vcap.set(cv::CAP_PROP_FRAME_HEIGHT, 480) ||
            !vcap.set(cv::CAP_PROP_FRAME_WIDTH, 640) ||
            !vcap.set(cv::CAP_PROP_FPS, 30)) {
            std::cout << "Setting vcap Failed\n";
            cv::Mat simpleframe;
            if (!vcap.read(simpleframe)) {
                settings_success = false;
            }
        }
    } catch (...) {
        std::cout << "Exception occurred while setting properties\n";
        settings_success = false;
    }

    return settings_success;
}

// Tries to open the camera with custom settings set in configureCamera
// If not possible, open it with default settings
bool openCamera(cv::VideoCapture& vcap, int device, int api) {
    vcap.open(device, api);
    if (!vcap.isOpened()) {
        std::cerr << "Failed to open vcap\n";
        return false;
    }

    if (!configureCamera(vcap)) {
        vcap.release();
        vcap.open(device, api);
        if (vcap.isOpened()) {
            std::cout << "Reopened vcap with original resolution\n";
        } else {
            std::cerr << "Failed to reopen vcap\n";
            return false;
        }
    }

    return true;
}

class DepthEstimation{
    private:
        MX::Types::MxModelInfo model_info;
        cv::VideoCapture vcap;
        int origHeight;
        int origWidth;
        int model_input_height;
        int model_input_width;
        int model_output_height;
        int model_output_width;

        //FPS calculation variables
        int frame_count;
        float fps_number;
        std::string fps_text;
        std::chrono::milliseconds start_ms;

        std::string window_name;
        bool window_created;
        cv::Size displaySize;

        //incallback images
        cv::Mat img_resized;
        cv::Mat img_model_in;

        //outcallback images
        cv::Mat img_model_out_uint;
        cv::Mat img_final_output;
        cv::Mat img_final_out_resized;
        cv::Mat img_model_out;
    public:
        DepthEstimation(MX::Runtime::MxAccl* accl, bool use_cam)
        {
            model_info = accl->get_model_info(0);
            model_input_height = model_info.in_featuremap_shapes[0][0];
            model_input_width = model_info.in_featuremap_shapes[0][1];
            model_output_height = model_info.out_featuremap_shapes[0][0];
            model_output_width = model_info.out_featuremap_shapes[0][1];
            if(use_cam){
                #ifdef __linux__
                    std::cout << "Running on Linux" << "\n";
                    if (!openCamera(vcap, 0, cv::CAP_V4L2)) {
                        throw(std::runtime_error("Failed to open: camera 0"));
                    }

                #elif defined(_WIN32)
                    std::cout << "Running on Windows" << "\n";
                    if (!openCamera(vcap, 0, cv::CAP_ANY)) {
                        throw(std::runtime_error("Failed to open: camera 0"));
                    }
                #endif
            }
            else{
                std::cout << "Running on Path: " << videoPath.string() << "\n";
                vcap.open(videoPath.string(), cv::CAP_ANY);
            }

            if(vcap.isOpened()){
                std::cout << "videocapture opened \n";

                origWidth = vcap.get(cv::CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
                origHeight = vcap.get(cv::CAP_PROP_FRAME_HEIGHT);
                runflag.store(true);
            }
            else{
                std::cout << "videocapture NOT opened \n";
                runflag.store(false);
            }

            frame_count = 0;
            fps_number = 0;
            fps_text = "FPS = ";
            start_ms = std::chrono::milliseconds(0);

            auto in_cb = std::bind(&DepthEstimation::incallback_getframe, this, std::placeholders::_1, std::placeholders::_2);
            auto out_cb = std::bind(&DepthEstimation::outcallback_getmxaoutput, this, std::placeholders::_1, std::placeholders::_2);
            accl->connect_stream(in_cb, out_cb, 0, 0);
            window_name = "Depth Estimation";
            window_created = false;

            img_resized.create(model_input_height, model_input_width, CV_8UC3);
            img_model_in.create(model_input_height, model_input_width, CV_32FC3);

            int display_width = (origWidth < 1920) ? origWidth : 1920;
            int display_height = (origHeight < 1080) ? origHeight : 1080;
            displaySize = cv::Size(display_width, display_height);
            img_model_out.create(cv::Size(model_output_height, model_output_width), CV_32FC1);
            img_model_out_uint.create(img_model_out.size(), CV_8UC1);
            img_final_output.create(img_model_out.size(), CV_8UC1);
            img_final_out_resized.create(displaySize, CV_8UC3);
        }

        // Input callback function
        bool incallback_getframe(std::vector<const MX::Types::FeatureMap*> dst, int streamLabel){
            if(runflag.load()){
                cv::Mat inframe;
                bool got_frame = vcap.read(inframe);

                if (!got_frame) {
                    std::cout << "No frame \n\n\n";
                    runflag.store(false);
                    return false;
                }
                else{
                    cv::resize(inframe, img_resized, img_resized.size());
                    cv::cvtColor(img_resized, img_resized, cv::COLOR_BGR2RGB);
                    img_resized.convertTo(img_model_in, CV_32FC3, 1.0 / 255.0);
                    cv::add(img_model_in, cv::Scalar(-0.485, -0.456, -0.406), img_model_in);
                    cv::multiply(img_model_in, cv::Scalar(1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225), img_model_in);

                    dst[0]->set_data((float*)img_model_in.data);
                    return true;
                }
            }
            else{
                vcap.release();
                return false;
            }
        }

        // Output callback function
        bool outcallback_getmxaoutput(std::vector<const MX::Types::FeatureMap*> src, int streamLabel){
            src[0]->get_data((float*)img_model_out.data);

            double depth_min_d, depth_max_d;
            float depth_min, depth_max;
            cv::minMaxIdx(img_model_out, &depth_min_d, &depth_max_d);
            depth_min = (float)depth_min_d;
            depth_max = (float)depth_max_d;
            float diff = depth_max - depth_min;

            cv::add(img_model_out, cv::Scalar(-depth_min), img_model_out);
            cv::multiply(img_model_out, cv::Scalar(1.0 / diff), img_model_out);
            cv::multiply(img_model_out, cv::Scalar(255.0), img_model_out);

            img_model_out.convertTo(img_model_out_uint, CV_8UC1);
            cv::applyColorMap(img_model_out_uint, img_final_output, cv::COLORMAP_INFERNO);

            frame_count++;
            if (frame_count == 1)
            {
                start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
            }
            else if (frame_count % AVG_FPS_CALC_FRAME_COUNT == 0)
            {
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()) - start_ms;
                fps_number = (float)AVG_FPS_CALC_FRAME_COUNT * 1000 / (float)(duration.count());

                // Round to 1 decimal and manually truncate after rounding
                fps_number = std::round(fps_number * 10.0f) / 10.0f;
                fps_text = "FPS = " + std::to_string(fps_number).substr(0, std::to_string(fps_number).find('.') + 2);

                frame_count = 0;
            }

            cv::resize(img_final_output, img_final_out_resized, displaySize);
            cv::putText(img_final_out_resized, fps_text,
                        cv::Point2i(10, 30), cv::FONT_ITALIC, 0.8,
                        cv::Scalar(255, 255, 0), 2);

            if (!window_created){
                cv::namedWindow(window_name, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
                cv::resizeWindow(window_name, displaySize);
                window_created = true;
            }

            cv::imshow(window_name, img_final_out_resized);
            if (cv::waitKey(1) == 'q') {
                runflag.store(false);
            }
            return true;
        }
        ~DepthEstimation(){
        }
};

int main(int argc, char* argv[]){
    bool use_cam = true;  // Default to using camera
    std::string dfpPath = "./midas_v2_small.dfp";  // Default DFP path (local to the binary)

    if(argc > 1) {
        for(int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if(arg == "--video") {
                use_cam = false;
                if(i + 1 < argc) {
                    videoPath = argv[++i];
                } else {
                    std::cerr << "Error: Missing video path after --video\n";
                    return 1;
                }
            } else if(arg == "-d") {
                if(i + 1 < argc) {
                    dfpPath = argv[++i];
                } else {
                    std::cerr << "Error: Missing DFP path after -d\n";
                    return 1;
                }
            }
        }
    }

    signal(SIGINT, signalHandler);

    if(runflag.load()){
        std::cout << "Application start\n";
        std::cout << "Model path: " << dfpPath << "\n";
        if(!use_cam) {
            std::cout << "Video File is used as input\nVideo Path: " << videoPath.c_str() << "\n";
        }
        if (!fs::exists(dfpPath)) {
            std::cerr << "Error: Model file not found at path: " << dfpPath << std::endl;
            std::exit(EXIT_FAILURE);
        }

        #ifdef _WIN32
            // Change server address 
            server_addr = "localhost";
        #endif

        MX::Runtime::MxAccl accl(
            fs::path(dfpPath),                      // DFP path
            std::vector<int>{0},                    // device_ids_to_use
            std::array<bool, 2>{true, true},        // use_model_shape
            false,                                  // local_mode
            MX::RPC::SchedulerOptions{600, 0, false, 16, 12},  // sched_options
            MX::RPC::ClientOptions{false, 0},       // client_options
            server_addr,                            // server_addr
            10000,                                  // server_port_base
            false                                   // ignore_server_
        );

        DepthEstimation app(&accl, use_cam);
        accl.start();
        accl.wait();
        accl.stop();
    } else {
        std::cout << "App exiting without execution\n\n\n";
    }

    return 0;
}