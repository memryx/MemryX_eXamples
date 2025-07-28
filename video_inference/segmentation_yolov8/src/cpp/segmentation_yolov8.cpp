#include <iostream>
#include <thread>
#include <signal.h>
#include <opencv2/opencv.hpp>    /* imshow */
#include <opencv2/imgproc.hpp>   /* cvtcolor */
#include <opencv2/imgcodecs.hpp> /* imwrite */
#include <chrono>
#include "memx/accl/MxAccl.h"
#include <memx/mxutils/gui_view.h>
#include <filesystem>
#include <queue>
namespace fs = std::filesystem;

std::atomic_bool runflag; // Atomic flag to control run state
bool src_is_cam = true;   // Flag to determine whether to use the camera or video
bool use_tflite = false;  // Flag to determine whether to use ONNX or TFLite

// YoloV8 application specific parameters
fs::path model_path = "models/onnx/YOLO_v8_nano_seg_640_640_3_onnx.dfp";                      // Default model path
fs::path postprocessing_model_path = "models/onnx/YOLO_v8_nano_seg_640_640_3_onnx_post.onnx"; // Default post-processing model path

#define AVG_FPS_CALC_FRAME_COUNT 50 // Number of frames used to calculate average FPS
#define FRAME_QUEUE_MAX_LENGTH 5    // Maximum number of frames to store in the queue

// Signal handler to gracefully stop the program on SIGINT (Ctrl+C)
void signal_handler(int p_signal)
{
    runflag.store(false); // Stop the program
}

// Function to display usage information
void printUsage(const std::string &programName)
{
    std::cout << "Usage: " << programName
              << " [-d <dfp_path>] [-p <post_model>] [--video \"video_path\"]\n"
              << "Options:\n"
              << "  -d, --dfp_path        (Optional) Path to the DFP. Default: " << model_path << "\n"
              << "  -p, --post_model      (Optional) Path to the post-model. Default: " << postprocessing_model_path << "\n"
              << "  --video               (Optional) Video path.  Default: use cam as input\n";
}

// Struct to store detected bounding boxes and related info
struct Box
{
    int x1, y1, x2, y2, class_id;
    float confidence;
    Box(int x1_, int y1_, int x2_, int y2_, int class_id_, float confidence_)
        : x1(x1_), y1(y1_), x2(x2_), y2(y2_), class_id(class_id_), confidence(confidence_) {}
};

// Function to configure camera settings (resolution and FPS)
bool configureCamera(cv::VideoCapture &vcap)
{
    bool settings_success = true;
    try
    {
        // Attempt to set 640x480 resolution and 30 FPS
        if (!vcap.set(cv::CAP_PROP_FRAME_HEIGHT, 480) ||
            !vcap.set(cv::CAP_PROP_FRAME_WIDTH, 640) ||
            !vcap.set(cv::CAP_PROP_FPS, 30))
        {
            std::cout << "Setting vcap Failed\n";
            cv::Mat simpleframe;
            if (!vcap.read(simpleframe))
            {
                settings_success = false;
            }
        }
    }
    catch (...)
    {
        std::cout << "Exception occurred while setting properties\n";
        settings_success = false;
    }
    return settings_success;
}

// Function to open the camera and apply settings, if not possible, reopen with default settings
bool openCamera(cv::VideoCapture &vcap, int api, int device = 0)
{
    vcap.open(device, api); // Open the camera
    if (!vcap.isOpened())
    {
        std::cerr << "Failed to open vcap\n";
        return false;
    }

    if (!configureCamera(vcap))
    {                   // Try applying custom settings
        vcap.release(); // Release and reopen with default settings
        vcap.open(device, api);
        if (vcap.isOpened())
        {
            std::cout << "Reopened vcap with original resolution\n";
        }
        else
        {
            std::cerr << "Failed to reopen vcap\n";
            return false;
        }
    }
    return true;
}

class YoloV8
{
private:
    // Model Params
    int model_input_width;      // width of model input image
    int model_input_height;     // height of model input image
    int ori_image_width;        // width of original input frame
    int ori_image_height;       // height of original input frame
    int num_boxes = 8400;       // YOLOv8 has 8400 anchor points
    int features_per_box = 84;  // number of output features per box
    int features_per_mask = 32; // number of output features per mask
    int num_features = features_per_box + features_per_mask;
    int mask_proto_width = 160;  // width of mask prototype
    int mask_proto_height = 160; // height of mask prototype
    float conf_thresh = 0.3;     // Confidence threshold
    float nms_thresh = 0.5;      // Non-maximum suppression threshold
    float maskThreshold = 0.5;
    std::vector<std::string> class_names = { // Class labels for COCO dataset
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "sofa", "potted plant", "bed", "dining table", "toilet", "tv monitor", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
    int num_class = class_names.size();
    std::vector<cv::Scalar> colors;

    // Application Variables
    std::queue<cv::Mat> frames_queue; // Queue for frames
    std::mutex frame_queue_mutex;     // Mutex to control access to the queue
    int frame_count = 0;
    float fps_number = .0; // FPS counter
    std::chrono::milliseconds start_ms;
    cv::VideoCapture vcap;                  // Video capture object
    MX::Types::MxModelInfo model_info;      // Model info structure
    MX::Types::MxModelInfo post_model_info; // Model info structure

    float *mxa_outputs[2] = {nullptr, nullptr}; // Buffer for the output of the accelerator
    MxQt *gui_;                                 // GUI for display
    cv::Mat displayImage;

    float scale_ratio;
    int pad_w, pad_h;

    // Function to preprocess the input image (resize and normalize)
    cv::Mat preprocess(cv::Mat &image)
    {
        // Compute resize ratio
        cv::Size new_unpad(static_cast<int>(ori_image_width * scale_ratio),
                           static_cast<int>(ori_image_height * scale_ratio));

        // Compute padding
        pad_w = (model_input_width - new_unpad.width) / 2;
        pad_h = (model_input_height - new_unpad.height) / 2;

        // Resize the image
        cv::Mat resized;
        cv::resize(image, resized, new_unpad, 0, 0, cv::INTER_LINEAR);

        // Create a padded image with a default value of 114 (same as original)
        cv::Mat padded_img(model_input_height, model_input_width, CV_8UC3, cv::Scalar(114, 114, 114));

        // Copy the resized image into the center of the padded image
        resized.copyTo(padded_img(cv::Rect(pad_w, pad_h, new_unpad.width, new_unpad.height)));

        // Normalize the image to [0,1]
        cv::Mat img_float;
        padded_img.convertTo(img_float, CV_32F, 1.0 / 255.0);

        return img_float;
    }

    cv::Mat get_mask(const cv::Mat &mask_coef,
                     const cv::Mat &mask_protos)
    {
        // Compute the mask by multiplying coefficients with prototypical masks
        // [1, 32] * [32, 160 * 160] => [1, 160 * 160]
        cv::Mat matmul = mask_coef * mask_protos;

        // [1, 160 * 160] => [160, 160]
        cv::Mat masks = matmul.reshape(1, {mask_proto_width, mask_proto_height});

        // sigmoid
        cv::Mat dest;
        cv::exp(-masks, dest);
        dest = 1.0 / (1.0 + dest);

        // resize
        cv::resize(dest, dest, cv::Size(model_input_width, model_input_height), cv::INTER_LINEAR);
        return dest;
    }

    // Function to process model output and get bounding boxes
    void get_detections(float *ofmap, int num_boxes, const cv::Mat &inframe,
                        std::vector<Box> &filtered_boxes,
                        std::vector<std::vector<float>> &filtered_mask_coefs)
    {
        std::vector<Box> all_boxes;
        std::vector<cv::Rect> cv_boxes;
        std::vector<float> all_scores;
        std::vector<std::vector<float>> mask_coefs;

        // [116,8400] => [8400,116]
        cv::Mat ofmap_t = cv::Mat(num_features, num_boxes, CV_32F, ofmap).t();
        float *ofmap_t_ptr = (float *)ofmap_t.data;

        // Iterate through the model outputs
        for (int i = 0; i < num_boxes; ++i)
        {
            float *feature_ptr = ofmap_t_ptr + i * num_features;

            // get best class information
            float confidence;
            int class_id;
            get_best_class_info(feature_ptr, confidence, class_id);

            if (confidence > conf_thresh)
            {
                // Extract and scale the bounding box coordinates
                float x0 = feature_ptr[0];
                float y0 = feature_ptr[1];
                float w = feature_ptr[2];
                float h = feature_ptr[3];

                // Convert boxes from center format (cxcywh) to corner format (xyxy)
                int x1 = static_cast<int>(x0 - 0.5f * w);
                int y1 = static_cast<int>(y0 - 0.5f * h);
                int x2 = x1 + w;
                int y2 = y1 + h;

                // Rescale boxes to original image size
                x1 = (x1 - pad_w) / scale_ratio;
                x2 = (x2 - pad_w) / scale_ratio;
                y1 = (y1 - pad_h) / scale_ratio;
                y2 = (y2 - pad_h) / scale_ratio;

                // Clamp box boundaries to image dimensions
                x1 = std::max(0, std::min(x1, ori_image_width - 1));
                x2 = std::max(0, std::min(x2, ori_image_width - 1));
                y1 = std::max(0, std::min(y1, ori_image_height - 1));
                y2 = std::max(0, std::min(y2, ori_image_height - 1));

                // Add detected box to the list
                all_boxes.emplace_back(x1, y1, x2, y2, class_id, confidence);
                all_scores.emplace_back(confidence);
                cv_boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);

                // Add detected mask coefficient to the list
                std::vector<float> mask_coef(feature_ptr + 4 + num_class, feature_ptr + num_features);
                mask_coefs.push_back(mask_coef);
            }
        }

        // Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
        std::vector<int> nms_result;
        if (!cv_boxes.empty())
        {
            cv::dnn::NMSBoxes(cv_boxes, all_scores, conf_thresh, nms_thresh, nms_result);
            for (int idx : nms_result)
            {
                filtered_boxes.push_back(all_boxes[idx]);
                filtered_mask_coefs.push_back(mask_coefs[idx]);
            }
        }
    }

    void get_best_class_info(float *feature, float &best_conf, int &best_classid)
    {
        // first 4 element are box
        best_classid = 4;
        best_conf = 0;

        for (int i = 4; i < num_class + 4; i++)
        {
            if (feature[i] > best_conf)
            {
                best_conf = feature[i];
                best_classid = i - 4;
            }
        }
    }

    cv::Mat scale_mask(const cv::Mat &mask, const float maskThreshold)
    {
        // Make sure the mask is within the image boundary
        int top = int(round(pad_h - 0.1));
        int left = int(round(pad_w - 0.1));
        int bottom = int(round(model_input_height - pad_h + 0.1));
        int right = int(round(model_input_width - pad_w + 0.1));

        // Crop the mask using calculated indices
        cv::Mat cropped_mask = mask(cv::Range(top, bottom), cv::Range(left, right));

        // mask = mask(cv::Rect(pad[0], pad[1], resized_shape.width - 2 * pad[0], resized_shape.height - 2 * pad[1]));

        cv::Mat scale_mask;
        cv::resize(cropped_mask, scale_mask, {ori_image_width, ori_image_height}, cv::INTER_LINEAR);

        return scale_mask;
    }

    // Function to draw bounding boxes on the image
    void draw_bounding_boxes(cv::Mat &image, const std::vector<Box> &boxes,
                             const std::vector<cv::Mat> &masks, float maskThreshold)
    {
        cv::Mat overlay = image.clone();
        for (int i = 0; i < boxes.size(); i++)
        {
            const Box &box = boxes.at(i);
            cv::Mat mask = masks.at(i).clone();

            // Draw bounding box
            cv::Rect box_rect = cv::Rect(cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2));
            cv::rectangle(overlay, box_rect, cv::Scalar(0, 255, 0), 2);

            cv::Scalar color = colors[box.class_id];
            cv::Mat tmp = cv::Mat::zeros(mask.size(), mask.type());

            // only keep the mask within bounding box
            cv::rectangle(tmp, box_rect, 1, cv::FILLED);
            cv::multiply(mask, tmp, mask);

            // Find and draw mask contours
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask > maskThreshold, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            cv::drawContours(overlay, contours, -1, color, cv::FILLED);            // Fill mask color
            cv::drawContours(overlay, contours, -1, cv::Scalar(255, 255, 255), 2); // White border line

            // Format class label with confidence score
            std::ostringstream labelStream;
            labelStream << class_names[box.class_id] << ": " << std::fixed << std::setprecision(2) << box.confidence;
            std::string label = labelStream.str();

            // Calculate label background size
            int baseline = 0;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

            // Draw label above bounding box
            cv::putText(overlay, label, cv::Point(box.x1, box.y1 - labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }
        // Blend the overlay with the original image
        cv::addWeighted(image, 0.3, overlay, 0.7, 0, image);
    }

    // Input callback function to fetch frames and preprocess them
    bool incallback_getframe(vector<const MX::Types::FeatureMap *> dst, int streamLabel)
    {
        if (runflag.load())
        {
            cv::Mat inframe;
            cv::Mat rgbImage;

            while (true)
            {

                bool got_frame = vcap.read(inframe); // Capture frame

                if (!got_frame)
                { // If no frame, stop the stream
                    std::cout << "No frame \n\n\n";
                    vcap.release();
                    return false;
                }

                if (src_is_cam && (frames_queue.size() >= FRAME_QUEUE_MAX_LENGTH))
                {
                    // drop the frame and try again if we've hit the limit
                    continue;
                }
                else
                {
                    // Convert frame to RGB and store in queue
                    cv::cvtColor(inframe, rgbImage, cv::COLOR_BGR2RGB);
                    {
                        std::lock_guard<std::mutex> ilock(frame_queue_mutex);
                        frames_queue.push(rgbImage);
                    }

                    // Preprocess frame and set data for inference
                    cv::Mat preProcframe = preprocess(rgbImage);
                    if(!use_tflite) {
                        cv::Mat chw_image;
                        cv::dnn::blobFromImage(preProcframe, chw_image, 1.0, cv::Size(model_input_width, model_input_height), cv::Scalar(0,0,0), true, false);
                        preProcframe = chw_image;
                    }
                    dst[0]->set_data((float *)preProcframe.data);

                    return true;
                }
            }
        }
        else
        {
            vcap.release();
            return false;
        }
    }

    // Output callback function to process MXA output and display results
    bool outcallback_getmxaoutput(vector<const MX::Types::FeatureMap *> src, int streamLabel)
    {
        // Get data from the feature maps
        for (int i = 0; i < post_model_info.num_out_featuremaps; ++i)
            src[i]->get_data(mxa_outputs[i]); // Get the output data from MXA

        {
            std::lock_guard<std::mutex> ilock(frame_queue_mutex);
            displayImage = frames_queue.front();
            frames_queue.pop();
        }
        // Get detected boxes and draw them on the image
        std::vector<std::vector<float>> mask_coefs;
        std::vector<Box> detected_boxes;
        get_detections(mxa_outputs[1], num_boxes, displayImage,
                       detected_boxes, mask_coefs);

        // Init mask prototypes
        cv::Mat mask_protos;

        if (use_tflite)
        {
            // [160*160, 32]
            mask_protos = cv::Mat(mask_proto_height * mask_proto_width, features_per_mask, CV_32F, mxa_outputs[0]);
            // [160*160, 32] => [32, 160*160]
            mask_protos = mask_protos.t();
        }
        else // use ONNX
        {
            // [32, 160*160]
            mask_protos = cv::Mat(features_per_mask, mask_proto_height * mask_proto_width, CV_32F, mxa_outputs[0]);
            // mask_protos = mask_protos.t();
        }

        // postprocess masks (crop & scale)
        std::vector<cv::Mat> detected_masks;
        for (int i = 0; i < detected_boxes.size(); i++)
        {
            // [32, 1] => [1, 32]
            cv::Mat mask_coef = cv::Mat(mask_coefs[i]).t();
            cv::Mat mask = get_mask(mask_coef, mask_protos);

            // scale mask to original image size
            cv::Mat scaled = scale_mask(mask, maskThreshold);
            detected_masks.push_back(scaled);
        }

        draw_bounding_boxes(displayImage, detected_boxes, detected_masks, maskThreshold);

        // Display the updated image in the GUI
        gui_->screens[0]->SetDisplayFrame(streamLabel, &displayImage, fps_number);

        // Calculate FPS
        frame_count++;
        if (frame_count == 1)
        {
            start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        }
        else if (frame_count % AVG_FPS_CALC_FRAME_COUNT == 0)
        {
            std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()) - start_ms;
            fps_number = (float)AVG_FPS_CALC_FRAME_COUNT * 1000 / (float)(duration.count());
            frame_count = 0;
        }

        return true;
    }

public:
    // Constructor to initialize YOLOv8 object
    YoloV8(MX::Runtime::MxAccl *accl, std::string video_src, MxQt *gui, int index)
    {
        gui_ = gui;

        // Open the camera or video source
        if (src_is_cam)
        {
            std::cout << "Use cam input...\n";
#ifdef __linux__
            if (!openCamera(vcap, cv::CAP_V4L2))
            {
                throw(std::runtime_error("Failed to open: " + video_src));
            }
#elif defined(_WIN32)
            if (!openCamera(vcap, cv::CAP_ANY))
            {
                throw(std::runtime_error("Failed to open: " + video_src));
            }
#endif
        }
        else
        {
            std::cout << "Video source given = " << video_src << "\n\n";
            vcap.open(video_src, cv::CAP_ANY);
        }

        if (!vcap.isOpened())
        {
            std::cout << "videocapture for " << video_src << " is NOT opened\n";
            runflag.store(false);
            exit(1);
        }

        // Init color
        for (int i = 0; i < num_class; i++)
        {
            int b = rand() % 256;
            int g = rand() % 256;
            int r = rand() % 256;
            colors.push_back(cv::Scalar(b, g, r));
        }

        // Get input image dimensions
        ori_image_width = static_cast<int>(vcap.get(cv::CAP_PROP_FRAME_WIDTH));
        ori_image_height = static_cast<int>(vcap.get(cv::CAP_PROP_FRAME_HEIGHT));

        // Get model info and allocate output buffer
        model_info = accl->get_model_info(0);
        post_model_info = accl->get_post_model_info(0);

        // Allocate memory for model output
        mxa_outputs[0] = new float[features_per_mask * mask_proto_height * mask_proto_width];
        mxa_outputs[1] = new float[num_boxes * (features_per_box + features_per_mask)];

        // Get model input dimensions
        model_input_height = model_info.in_featuremap_shapes[0][0];
        model_input_width = model_info.in_featuremap_shapes[0][1];

        scale_ratio = std::min(static_cast<double>(model_input_height) / ori_image_height,
                               static_cast<double>(model_input_width) / ori_image_width);

        // Bind input/output callback functions
        auto in_cb = std::bind(&YoloV8::incallback_getframe, this, std::placeholders::_1, std::placeholders::_2);
        auto out_cb = std::bind(&YoloV8::outcallback_getmxaoutput, this, std::placeholders::_1, std::placeholders::_2);

        // Connect streams to the accelerator
        accl->connect_stream(in_cb, out_cb, index /**Unique Stream Idx */, 0 /**Model Idx */);

        // Start the input/output streams
        runflag.store(true);
    }

    ~YoloV8()
    {
        for (int i = 0; i < 2; ++i)
        {
            delete[] mxa_outputs[i]; // Clean up memory
            mxa_outputs[i] = nullptr;
        }
    }
};

int main(int argc, char *argv[])
{
    signal(SIGINT, signal_handler); // Set up signal handler
    vector<string> video_src_list;

    std::string video_str = "cam";

    // Iterate through the arguments
    for (int i = 1; i < argc; i++)
    {

        std::string arg = argv[i];

        // Handle -d or --dfp_path
        if (arg == "-d" || arg == "--dfp_path")
        {
            if (i + 1 < argc && argv[i + 1][0] != '-')
            { // Ensure there's a next argument and it is not another option
                model_path = argv[++i];
            }
            else
            {
                std::cerr << "Error: Missing value for " << arg << " option.\n";
                printUsage(argv[0]);
                return 1;
            }
        }
        // Handle -m or --post_model
        else if (arg == "-p" || arg == "--post_model")
        {
            if (i + 1 < argc && argv[i + 1][0] != '-')
            { // Ensure there's a next argument and it is not another option
                postprocessing_model_path = argv[++i];
            }
            else
            {
                std::cerr << "Error: Missing value for " << arg << " option.\n";
                printUsage(argv[0]);
                return 1;
            }
        }
        // Handle --video_paths
        else if (arg == "--video")
        {
            src_is_cam = false; // Use video
            if (i + 1 < argc && argv[i + 1][0] != '-')
            { // Ensure there's a next argument and it is not another option
                video_str = std::filesystem::path(argv[++i]);
            }
            else
            {
                std::cerr << "Error: Missing value for " << arg << " option.\n";
                printUsage(argv[0]);
                return 1;
            }
        }
        // Handle unknown options
        else
        {
            std::cerr << "Error: Unknown option " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    // Check if the post-processing model is a TFLite model
    use_tflite = postprocessing_model_path.extension() == ".tflite";

    // Create the Accl object and load the DFP model
    MX::Runtime::MxAccl* accl;
    accl = new MX::Runtime::MxAccl(model_path);

    // Connect the post-processing model
    accl->connect_post_model(postprocessing_model_path);

    // Creating GuiView for display
    MxQt gui(argc, argv);
    gui.screens[0]->SetSquareLayout(1, false); // Single stream layout

    // Creating YoloV8 objects for each video stream
    YoloV8 obj(accl, video_str, &gui, 0);

    // Run the accelerator and wait
    accl->start();
    gui.Run(); // Wait until the exit button is pressed in the Qt window
    accl->stop();
}
