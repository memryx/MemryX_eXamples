#include <iostream>
#include <thread>
#include <signal.h>
#include <opencv2/opencv.hpp>    /* imshow */
#include <opencv2/imgproc.hpp>   /* cvtcolor */
#include <opencv2/imgcodecs.hpp> /* imwrite */
#include <chrono>
#include "memx/accl/MxAccl.h"
#include <memx/mxutils/gui_view.h>

namespace fs = std::filesystem;

std::atomic_bool runflag;  // Atomic flag to control run state

// YoloV8 application specific parameters
fs::path model_path = "YOLO_v8_small_640_640_3_tflite.dfp";  // Default model path
fs::path postprocessing_model_path = "YOLO_v8_small_640_640_3_tflite_post.tflite";  // Default post-processing model path
#define AVG_FPS_CALC_FRAME_COUNT  50  // Number of frames used to calculate average FPS
#define FRAME_QUEUE_MAX_LENGTH     5

// Signal handler to gracefully stop the program on SIGINT (Ctrl+C)
void signal_handler(int p_signal) {
    runflag.store(false);  // Stop the program
}

// Function to display usage information
void printUsage(const std::string& programName) {
    std::cout << "Usage: " << programName
              << " [-d <dfp_path>] [-m <post_model>] [--video_paths \"cam:0,vid:video_path\"]\n"
              << "Options:\n"
              << "  -d, --dfp_path        (Optional) Path to the DFP. Default: "<< model_path<<"\n"
              << "  -m, --post_model      (Optional) Path to the post-model. Default: "<<postprocessing_model_path<<"\n"
              << "  --video_paths         (Optional) Video paths in the format \"cam:0,vid:video_path,vid:video2_path\". Default: cam:0\n";
}

// Struct to store detected bounding boxes and related info
struct Box {
    float x1, y1, x2, y2, confidence, class_id;
};

// Function to configure camera settings (resolution and FPS)
bool configureCamera(cv::VideoCapture& vcap) {
    bool settings_success = true;
    try {
        // Attempt to set 640x480 resolution and 30 FPS
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

// Function to open the camera and apply settings, if not possible, reopen with default settings
bool openCamera(cv::VideoCapture& vcap, int device, int api) {
    vcap.open(device, api);  // Open the camera
    if (!vcap.isOpened()) {
        std::cerr << "Failed to open vcap\n";
        return false;
    }

    if (!configureCamera(vcap)) {  // Try applying custom settings
        vcap.release();  // Release and reopen with default settings
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

class YoloV8 {
    private:
        // Model Params
        int model_input_width;  // width of model input image
        int model_input_height; // height of model input image
        int input_image_width;  // width of input image
        int input_image_height; // height of input image
        int num_boxes = 8400;   // YOLOv8 has 8400 anchor points
        int features_per_box = 84; // number of output features per box
        float conf_thresh = 0.6;  // Confidence threshold
        float nms_thresh = 0.5;   // Non-maximum suppression threshold
        std::vector<std::string> class_names = {  // Class labels for COCO dataset
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "sofa", "potted plant", "bed", "dining table", "toilet", "tv monitor", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        };

        // Application Variables
        std::deque<cv::Mat> frames_queue;  // Queue for frames
        std::mutex frame_queue_mutex;  // Mutex to control access to the queue
        int num_frames = 0;
        int frame_count = 0;
        float fps_number = .0;  // FPS counter
        std::chrono::milliseconds start_ms;
        cv::VideoCapture vcap;  // Video capture object
        bool src_is_cam = false;
        std::vector<size_t> in_tensor_sizes;
        std::vector<size_t> out_tensor_sizes;
        MX::Types::MxModelInfo model_info;  // Model info structure
        float* mxa_output;  // Buffer for the output of the accelerator
        cv::Mat displayImage;
        MxQt* gui_;  // GUI for display
        int length;
        std::string model_type; 

        std::vector<Box> all_boxes;
        std::vector<float> all_scores;
        std::vector<cv::Rect> cv_boxes;

        // Function to preprocess the input image (resize and normalize)
        cv::Mat preprocess(cv::Mat& image) {
            // Get original image dimensions
            int img_height = image.rows;
            int img_width = image.cols;

            // Determine the size of the square image (longest side)
            length = std::max(img_height, img_width);

            // Create a black square image with the same number of channels
            cv::Mat squareImage = cv::Mat::zeros(cv::Size(length, length), image.type());

            // Copy the original image into the square image
            image.copyTo(squareImage(cv::Rect(0, 0, img_width, img_height)));

            // Resize to 640x640 for the model input
            cv::Mat resizedImage;
            if (model_type == "tflite") {
                cv::resize(squareImage, resizedImage, cv::Size(640, 640), cv::INTER_LINEAR);
            } else {
                cv::dnn::blobFromImage(squareImage, resizedImage, 1.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);
            }

            // Convert to float32 and normalize pixel values (0-1 range)
            resizedImage.convertTo(resizedImage, CV_32F, 1.0 / 255.0);

            // Convert HWC (Height, Width, Channels) to CHW (Channels, Height, Width)
            std::vector<cv::Mat> channels(3);
            cv::split(resizedImage, channels);
            cv::Mat chwImage;
            cv::merge(channels, chwImage);

            return chwImage;
        }

        // Function to process model output and get bounding boxes
        std::vector<Box> get_detections(float* ofmap, int num_boxes, const cv::Mat& inframe) {
            std::vector<Box> all_boxes;
            std::vector<cv::Rect> cv_boxes;
            std::vector<float> all_scores;
            std::vector<Box> filtered_boxes;

            // Precompute scaling factors once
            const float y_factor = static_cast<float>(length) / static_cast<float>(model_input_height);
            const float x_factor = static_cast<float>(length) / static_cast<float>(model_input_width);

            // Iterate through the model outputs
            for (int i = 0; i < num_boxes; ++i) {
                // Extract and scale the bounding box coordinates
                float x0 = ofmap[i];
                float y0 = ofmap[num_boxes + i];
                float w = ofmap[2 * num_boxes + i];
                float h = ofmap[3 * num_boxes + i];
                x0 *= x_factor;
                y0 *= y_factor;
                w *= x_factor;
                h *= y_factor;

                int x1 = static_cast<int>(x0 - 0.5f * w);
                int y1 = static_cast<int>(y0 - 0.5f * h);
                int x2 = static_cast<int>(x0 + 0.5f * w);
                int y2 = static_cast<int>(y0 + 0.5f * h);

                // Loop through classes and apply confidence threshold
                for (int j = 4; j < features_per_box; ++j) {
                    float confidence = ofmap[j * num_boxes + i];

                    if (confidence > conf_thresh) {
                        // Add detected box to the list
                        Box box;
                        box.x1 = x1;
                        box.y1 = y1;
                        box.x2 = x2;
                        box.y2 = y2;
                        box.class_id = j - 4;
                        box.confidence = confidence;

                        all_boxes.push_back(box);
                        all_scores.push_back(confidence);
                        cv_boxes.emplace_back(cv::Rect(x1, y1, x2 - x1, y2 - y1));
                    }
                }
            }

            // Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
            std::vector<int> nms_result;
            if (!cv_boxes.empty()) {
                cv::dnn::NMSBoxes(cv_boxes, all_scores, conf_thresh, nms_thresh, nms_result);
                for (int idx : nms_result) {
                    filtered_boxes.push_back(all_boxes[idx]);
                }
            }

            return filtered_boxes;
        }

        // Function to draw bounding boxes on the image
        void draw_bounding_boxes(cv::Mat& image, const std::vector<Box>& boxes) {
            for (const Box& box : boxes) {
                // Draw bounding box
                cv::rectangle(image, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2), cv::Scalar(0, 255, 0), 2);

                // Add class label and confidence score
                std::string label = class_names[box.class_id];
                int baseLine = 0;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                cv::putText(image, label, cv::Point(box.x1, box.y1 - labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            }
        }

        // Input callback function to fetch frames and preprocess them
        bool incallback_getframe(std::vector<const MX::Types::FeatureMap*> dst, int streamLabel) {
            if (runflag.load()) {
                cv::Mat inframe;
                cv::Mat rgbImage;

                while(true){
                    bool got_frame = vcap.read(inframe);  // Capture frame

                    if (!got_frame) {  // If no frame, stop the stream
                        std::cout << "No frame \n\n\n";
                        vcap.release();
                        return false;
                    }

                    if(src_is_cam && (frames_queue.size() >= FRAME_QUEUE_MAX_LENGTH)){
                        // drop the frame and try again if we've hit the limit
                        continue;
                    }
                    else{
                        // Convert frame to RGB and store in queue
                        cv::cvtColor(inframe, rgbImage, cv::COLOR_BGR2RGB);
                        {
                            std::lock_guard<std::mutex> ilock(frame_queue_mutex);
                            frames_queue.push_back(rgbImage);
                        }
                    }

                    // Preprocess frame and set data for inference
                    cv::Mat preProcframe = preprocess(rgbImage);
                    dst[0]->set_data((float*)preProcframe.data);
                    return true;
                }
            }
            else {
                vcap.release();
                return false;
            }
        }

        // Output callback function to process MXA output and display results
        bool outcallback_getmxaoutput(std::vector<const MX::Types::FeatureMap*> src, int streamLabel) {
            src[0]->get_data(mxa_output);  // Get the output data from MXA

            {
                std::lock_guard<std::mutex> ilock(frame_queue_mutex);
                displayImage = frames_queue.front();
                frames_queue.pop_front();
            }

            // Get detected boxes and draw them on the image
            std::vector<Box> detected_objectVector = get_detections(mxa_output, num_boxes, displayImage);
            draw_bounding_boxes(displayImage, detected_objectVector);

            // Display the updated image in the GUI
            gui_->screens[0]->SetDisplayFrame(streamLabel, &displayImage, fps_number);

            // Calculate FPS
            frame_count++;
            if (frame_count == 1) {
                start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
            }
            else if (frame_count % AVG_FPS_CALC_FRAME_COUNT == 0) {
                std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()) - start_ms;
                fps_number = (float)AVG_FPS_CALC_FRAME_COUNT * 1000 / (float)(duration.count());
                frame_count = 0;
            }
            return true;
        }

    public:
        // Constructor to initialize YOLOv8 object
        YoloV8(MX::Runtime::MxAccl* accl, std::string video_src, std::string model_type, MxQt* gui, int index) {
            gui_ = gui;
            this->model_type = model_type;

            // Open the camera or video source
            if(video_src.substr(0,3) == "cam") {
                src_is_cam = true;
                int device = std::stoi(video_src.substr(4));
                #ifdef __linux__
                    if (!openCamera(vcap, device, cv::CAP_V4L2)) {
                        throw(std::runtime_error("Failed to open: "+video_src));
                    }
                #elif defined(_WIN32)
                    if (!openCamera(vcap, device, cv::CAP_ANY)) {
                        throw(std::runtime_error("Failed to open: "+video_src));
                    }
                #endif
            } else if (video_src.substr(0,3) == "vid") {
                src_is_cam = false;
                std::cout << "Video source given = " << video_src.substr(4) << "\n\n";
                vcap.open(video_src.substr(4), cv::CAP_ANY);
            } else {
                throw(std::runtime_error("Given video src: " + video_src + " is invalid"));
            }

            if (!vcap.isOpened()) {
                std::cout << "videocapture for " << video_src << " is NOT opened\n";
                runflag.store(false);
            }

            // Get input image dimensions
            input_image_width = static_cast<int>(vcap.get(cv::CAP_PROP_FRAME_WIDTH));
            input_image_height = static_cast<int>(vcap.get(cv::CAP_PROP_FRAME_HEIGHT));

            // Get model info and allocate output buffer
            model_info = accl->get_model_info(0);
            mxa_output = new float[num_boxes * features_per_box];  // Allocate memory for model output

            // Get model input dimensions
            model_input_height = model_info.in_featuremap_shapes[0][0];
            model_input_width = model_info.in_featuremap_shapes[0][1];

            // Bind input/output callback functions
            auto in_cb = std::bind(&YoloV8::incallback_getframe, this, std::placeholders::_1, std::placeholders::_2);
            auto out_cb = std::bind(&YoloV8::outcallback_getmxaoutput, this, std::placeholders::_1, std::placeholders::_2);

            // Connect streams to the accelerator
            accl->connect_stream(in_cb, out_cb, index /**Unique Stream Idx */, 0 /**Model Idx */);

            // Start the input/output streams
            runflag.store(true);
        }

        ~YoloV8() {
            delete[] mxa_output;  // Clean up memory
            mxa_output = nullptr;
        }
};

int main(int argc, char* argv[]) {
    signal(SIGINT, signal_handler);  // Set up signal handler
    vector<string> video_src_list;

   std::string video_str = "cam:0";

    // Iterate through the arguments
    for (int i = 1; i < argc; i++) {

        std::string arg = argv[i];

        // Handle -d or --dfp_path
        if (arg == "-d" || arg == "--dfp_path") {
            if (i + 1 < argc && argv[i + 1][0] != '-') {  // Ensure there's a next argument and it is not another option
                model_path = argv[++i];
            } else {
                std::cerr << "Error: Missing value for " << arg << " option.\n";
                printUsage(argv[0]);
                return 1;
            }
        }
        // Handle -m or --post_model
        else if (arg == "-m" || arg == "--post_model") {
            if (i + 1 < argc && argv[i + 1][0] != '-') {  // Ensure there's a next argument and it is not another option
                postprocessing_model_path = argv[++i];
            } else {
                std::cerr << "Error: Missing value for " << arg << " option.\n";
                printUsage(argv[0]);
                return 1;
            }
        }
        // Handle --video_paths
        else if (arg == "--video_paths") {
            if (i + 1 < argc && argv[i + 1][0] != '-') {  // Ensure there's a next argument and it is not another option
                video_str = argv[++i];
                 size_t pos = 0;
                std::string token;
                std::string delimiter = ",";
                while ((pos = video_str.find(delimiter)) != std::string::npos) {
                    token = video_str.substr(0, pos);
                    video_src_list.push_back(token);
                    video_str.erase(0, pos + delimiter.length());
                }
                video_src_list.push_back(video_str);
            } else {
                std::cerr << "Error: Missing value for " << arg << " option.\n";
                printUsage(argv[0]);
                return 1;
            }
        }
        // Handle unknown options
        else {
            std::cerr << "Error: Unknown option " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    // if video_paths arg isn't passed - use default video string.
    if(video_src_list.size()==0){
        video_src_list.push_back(video_str);
    }  

    std::string model_type;
    std::string path_str = postprocessing_model_path.string();  // convert path to string

    if (path_str.size() >= 5 && path_str.substr(path_str.size() - 5) == ".onnx") {
        model_type = "onnx";
    } else if (path_str.size() >= 7 && path_str.substr(path_str.size() - 7) == ".tflite") {
        model_type = "tflite";
    } else {
        std::cerr << "Unsupported post-processing model format: " << path_str << std::endl;
        return 1;
    }

    // Create the Accl object and load the DFP model
    MX::Runtime::MxAccl accl{fs::path(model_path)};

    // Connect the post-processing model
    accl.connect_post_model(fs::path(postprocessing_model_path));

    // Creating GuiView for display
    MxQt gui(argc, argv);
    if (video_src_list.size() == 1)
        gui.screens[0]->SetSquareLayout(1, false);  // Single stream layout
    else
        gui.screens[0]->SetSquareLayout(static_cast<int>(video_src_list.size()));  // Multi-stream layout

    // Creating YoloV8 objects for each video stream
    std::vector<YoloV8*> yolo_objs;
    for (int i = 0; i < video_src_list.size(); ++i) {
        YoloV8* obj = new YoloV8(&accl, video_src_list[i], model_type,  &gui, i);
        yolo_objs.push_back(obj);
    }

    // Run the accelerator and wait
    accl.start();
    gui.Run();  // Wait until the exit button is pressed in the Qt window
    accl.stop();

    // Cleanup
    for (int i = 0; i < video_src_list.size(); ++i) {
        delete yolo_objs[i];
    }
}
