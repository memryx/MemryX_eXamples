#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <csignal>
#include <queue>
#include <condition_variable>
#include <opencv2/opencv.hpp>

#include "memx/accl/MxAccl.h"
#include <memx/mxutils/gui_view.h>
#include "PoseApp.h"
#include "CartoonApp.h"

// Frame queues and sync
std::queue<cv::Mat> cartoon_queue;
std::queue<cv::Mat> pose_queue;
std::mutex cartoon_mutex, pose_mutex;
std::condition_variable cartoon_cv, pose_cv;

// Runtime state
std::atomic_bool runflag(true);
cv::VideoCapture vcap;

// Signal handler to safely stop
void signalHandler(int signum) {
    runflag.store(false);
}

const size_t MAX_QUEUE_SIZE = 20;  // You can adjust this as needed

void frameProducer(cv::VideoCapture& cap, std::atomic_bool& runflag, bool is_video) {
    cv::Mat frame;

    while (runflag.load()) {
        if (!cap.read(frame)) {
            std::cerr << "[Producer] End of video or failed to read frame.\n";
            break;
        }

        {
            std::lock_guard<std::mutex> lock(cartoon_mutex);
            cartoon_queue.push(frame.clone());
            cartoon_cv.notify_one();
        }

        {
            std::lock_guard<std::mutex> lock(pose_mutex);
            pose_queue.push(frame.clone());
            pose_cv.notify_one();
        }
    }

    if (is_video) {
        std::cout << "[Producer] Waiting for queues to flush before exiting...\n";

        while (runflag.load()) {
            std::unique_lock<std::mutex> lock1(cartoon_mutex, std::defer_lock);
            std::unique_lock<std::mutex> lock2(pose_mutex, std::defer_lock);
            std::lock(lock1, lock2);

            if (cartoon_queue.empty() && pose_queue.empty())
                break;

            lock1.unlock();
            lock2.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        std::cout << "[Producer] Flush wait done or interrupted. Proceeding to shutdown...\n";
    }

    runflag.store(false);
    cartoon_cv.notify_all();
    pose_cv.notify_all();
}


int main(int argc, char* argv[]) {
    bool use_cam = false;
    std::string video_path;

    // Init GUI
    MxQt gui(argc, argv);
    gui.screens[0]->SetSquareLayout(2);  // Two screens: Cartoon + Pose

    // Parse input arguments
    if (argc >= 2) {
        std::string inputType(argv[1]);
        if (inputType == "--cam") {
            use_cam = true;
        } else if (inputType == "--video" && argc >= 3) {
            video_path = argv[2];
        } else {
            std::cout << "Usage: ./cartoon_pose [--cam | --video <path>]\n";
            return -1;
        }
    } else {
        std::cout << "Usage: ./cartoon_pose [--cam | --video <path>]\n";
        return -1;
    }

    signal(SIGINT, signalHandler);

    if (use_cam) {
        vcap.open(0, cv::CAP_V4L2);
    } else {
        vcap.open(video_path);
    }

    if (!vcap.isOpened()) {
        std::cerr << "[Main] Failed to open input source.\n";
        return -1;
    }

    // Accelerator options
    MX::RPC::SchedulerOptions sched_opts{
        30,   // frame_limit: Max frames before DFP swap
        0,    // time_limit (ms): No time-based swap limit
        false,// stop_on_empty: Keep DFP active even if input is empty
        20,   // ifmap_queue_size: Input queue capacity
        20    // ofmap_queue_size: Output queue capacity per client
    };

    MX::RPC::ClientOptions client_opts{
        true,  // smoothing: Enable FPS smoothing
        30.0f  // fps_target: Limit input pacing to 30 FPS
    };

    // Cartoon pipeline
    std::vector<int> cartoon_device = {0};
    MX::Runtime::MxAccl accl_cartoon("Facial_cartoonizer_512_512_3_onnx.dfp", cartoon_device, {false, false}, false, sched_opts, client_opts);
    CartoonApp cartoonApp(&accl_cartoon, &cartoon_queue, &cartoon_mutex, &cartoon_cv, &runflag, &gui, 0);

    // Pose pipeline
    std::vector<int> pose_device = {0};
    MX::Runtime::MxAccl accl_pose("YOLO_v8_small_pose_640_640_3_onnx.dfp", pose_device, {true, true}, false, sched_opts, client_opts);
    PoseApp poseApp(&accl_pose, "YOLO_v8_small_pose_640_640_3_onnx_post.onnx", &pose_queue, &pose_mutex, &pose_cv, &runflag, &gui, 1);

    // std::thread producer_thread(frameProducer, std::ref(vcap), std::ref(runflag));
    std::thread producer_thread(frameProducer, std::ref(vcap), std::ref(runflag), !use_cam);

    accl_pose.start();
    accl_cartoon.start();

    gui.Run();
    std::cout << "[Main] GUI exited. Shutting down..." << std::endl;
    runflag.store(false);

    accl_pose.wait();
    accl_cartoon.wait();
    accl_pose.stop();
    accl_cartoon.stop();

    producer_thread.join();

    return 0;
}
