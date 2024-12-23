#include <iostream>
#include <thread>
#include <signal.h>
#include <opencv2/opencv.hpp>    /* imshow */
#include <opencv2/imgproc.hpp>   /* cvtcolor */
#include <opencv2/imgcodecs.hpp> /* imwrite */
#include <chrono>
#include "memx/accl/MxAccl.h"
#include <memx/mxutils/gui_view.h>
#include <curses.h>


#define LPD_INPUT_TENSOR_WIDTH 320
#define LPD_INPUT_TENSOR_HEIGHT 240
#define LPD_INPUT_TENSOR_CHANNEL 3

#define LPD_OUTPUT_TENSOR_LOC 61390 // 4385x14
#define LPD_OUTPUT_TENSOR_CONF 8770 // 4385x2
#define LPD_OUTPUT_TENSOR_IOU 4385  // 4385x1

#define LPR_INPUT_TENSOR_WIDTH 168
#define LPR_INPUT_TENSOR_HEIGHT 48
#define LPR_INPUT_TENSOR_CHANNEL 3

#define STREAM_ALLOCATE_BUFFER 5

namespace fs = std::filesystem;

std::atomic_bool runflag; // Atomic flag to control run state

// ALPR application specific parameters
fs::path model_path = "multi_lpd_yunet_lpr_crnn.dfp";               // Default model path
fs::path postprocessing_model_path = "lpd_yunet_post.onnx"; // Default post-processing model path
#define AVG_FPS_CALC_FRAME_COUNT 50                                 // Number of frames used to calculate average FPS

// Signal handler to gracefully stop the program on SIGINT (Ctrl+C)
void signal_handler(int p_signal)
{
    runflag.store(false); // Stop the program
}

// Function to display usage information
void printUsage(const std::string &programName)
{
    std::cout << "Usage: " << programName
              << " [-d <dfp_path>] [-m <post_model>] [--video_paths \"cam:0,vid:video_path\"]\n";
}

// Struct to store detected bounding boxes and related info
struct Box
{
    float x1, y1, x2, y2, confidence, class_id;
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
bool openCamera(cv::VideoCapture &vcap, int device, int api)
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

class ALPR
{
private:
    struct SolutionManagerDualModel
    {
        MX::Utils::fifo_queue<std::vector<float *>> queue_input_data_0;
        MX::Utils::fifo_queue<std::vector<float *>> queue_input_data_1;
        MX::Utils::fifo_queue<std::vector<float *>> queue_output_data_0;
        MX::Utils::fifo_queue<std::vector<float *>> queue_output_data_1;
        MX::Utils::fifo_queue<cv::Mat *> queue_crop_img_buffer;
        MX::Utils::fifo_queue<cv::Mat *> queue_img_buffer;
        MX::Utils::fifo_queue<cv::Mat *> queue_tmp_img;
        MX::Utils::fifo_queue<cv::Mat *> queue_crop_img;
        MX::Utils::fifo_queue<cv::Mat *> queue_recog_img;
        MX::Types::MxModelInfo model_info_0;
        MX::Types::MxModelInfo post_model_info;
        MX::Types::MxModelInfo model_info_1;
    };

    class LPDYunetPrePost
    {
    private:
        std::vector<std::string> output_names = {"loc", "conf", "iou"};
        std::vector<std::vector<int>> min_sizes = {{10, 16, 24}, {32, 48}, {64, 96}, {128, 192, 256}};
        std::vector<int> steps = {8, 16, 32, 64};
        std::vector<float> variance = {0.1, 0.2};
        std::pair<int, int> input_size = {0, 0};
        std::vector<std::vector<float>> priors;
        float confidence_threshold = 0.8f;
        float nms_threshold = 0.3f;
        const int top_k = 5000;
        const int keep_top_k = 750;

    public:
        LPDYunetPrePost() {};
        ~LPDYunetPrePost() {};
        void preprocessing(void *rgb_buf, int width, int height, std::vector<float *> in_bufs)
        {
            cv::Mat orig_mat(height, width, CV_8UC3, rgb_buf);
            cv::Mat resized_mat;
            cv::resize(orig_mat, resized_mat, cv::Size(LPD_INPUT_TENSOR_WIDTH, LPD_INPUT_TENSOR_HEIGHT), 0, 0, cv::INTER_LINEAR);

            if (input_size.first == 0 || input_size.second == 0)
            {
                input_size.first = width;
                input_size.second = height;
                prior_gen();
            }

            float *prelayer_processed_input = in_bufs[0];
            for (int i = 0; i < (LPD_INPUT_TENSOR_WIDTH * LPD_INPUT_TENSOR_HEIGHT * LPD_INPUT_TENSOR_CHANNEL); i++)
                prelayer_processed_input[i] = (float)resized_mat.data[i];
        }
        std::vector<std::vector<float>> postprocessing(std::vector<float *> out_bufs)
        {
            // out_bufs[0] loc OUTPUT_TENSOR_LOC #61390
            // out_bufs[1] conf OUTPUT_TENSOR_LOC #8870
            // out_bufs[2] iou OUTPUT_TENSOR_IOU #4385

            auto dets = decode(out_bufs);

            // Prepare data for NMS
            std::vector<cv::Rect> boxes;
            std::vector<float> scores;
            for (const auto &det : dets)
            {
                boxes.emplace_back(det[0], det[1], det[2], det[3]);
                scores.push_back(det[8]);
            }

            // NMS
            std::vector<int> indices;
            cv::dnn::NMSBoxes(boxes, scores, confidence_threshold, nms_threshold, indices, 1.f, top_k);

            // Filter results
            std::vector<std::vector<float>> result;
            for (int idx : indices)
            {
                if (result.size() >= keep_top_k)
                    break;
                result.push_back(dets[idx]);
            }

            return result;
        }

        void draw_bounding_box(std::vector<std::vector<float>> dets, cv::Mat &image, cv::Mat &image_cropped)
        {
            cv::Scalar line_color(0, 255, 0); // Green in BGR
            cv::Scalar text_color(0, 0, 255); // Red in BGR
            int j = 0;
            // Draw bounding boxes
            for (const auto &det : dets)
            {
                std::vector<cv::Point> points;
                for (int i = 0; i < 8; i += 2)
                {
                    points.emplace_back(static_cast<int>(det[i]), static_cast<int>(det[i + 1]));
                }

                // Draw the border of license plate
                points[0].x = std::max(points[0].x - 10, 0);
                points[0].y = std::max(points[0].y - 10, 0);
                points[3].x = std::min(points[3].x + 10, image.cols);
                points[3].y = std::min(points[3].y + 10, image.rows);

                cv::Rect roi(points[0], points[3]);
                image_cropped.create(roi.size(), CV_8UC3);
                image(roi).copyTo(image_cropped);

                cv::rectangle(image, cv::Rect(points[0], points[3]), line_color, 1);
            }
        }

        void prior_gen()
        {
            int w = LPD_INPUT_TENSOR_WIDTH;  // input_size.first; //LPD_INPUT_TENSOR_WIDTH;
            int h = LPD_INPUT_TENSOR_HEIGHT; // input_size.second; //LPD_INPUT_TENSOR_HEIGHT;

            std::vector<std::pair<int, int>> feature_maps(5);
            feature_maps[0] = {(h + 1) / 2 / 2, (w + 1) / 2 / 2};
            for (int i = 1; i < 5; ++i)
            {
                feature_maps[i] = {feature_maps[i - 1].first / 2, feature_maps[i - 1].second / 2};
            }

            priors.clear();

            for (size_t k = 0; k < 4; ++k)
            {
                const auto &f = feature_maps[k + 1]; // Using feature_map_3th to feature_map_6th
                const auto &min_sizes_k = min_sizes[k];

                for (int i = 0; i < f.first; ++i)
                {
                    for (int j = 0; j < f.second; ++j)
                    {
                        for (int min_size : min_sizes_k)
                        {
                            float s_kx = static_cast<float>(min_size) / w;
                            float s_ky = static_cast<float>(min_size) / h;
                            float cx = (j + 0.5f) * steps[k] / static_cast<float>(w);
                            float cy = (i + 0.5f) * steps[k] / static_cast<float>(h);
                            priors.push_back({cx, cy, s_kx, s_ky});
                        }
                    }
                }
            }
        }

        std::vector<std::vector<float>> decode(std::vector<float *> out_bufs)
        {
            float *loc = out_bufs[0];
            float *conf = out_bufs[1];
            float *iou = out_bufs[2];

            // Get number of elements
            size_t num_elements = LPD_OUTPUT_TENSOR_IOU;

            std::vector<float> cls_scores;
            std::vector<float> iou_scores;
            cls_scores.resize(num_elements);
            iou_scores.resize(num_elements);

            // Get scores
            for (size_t i = 0; i < num_elements; ++i)
            {
                cls_scores[i] = conf[i * 2 + 1];
                iou_scores[i] = iou[i];
            }

            // Clamp iou_scores
            for (auto &score : iou_scores)
            {
                score = std::max(0.0f, std::min(1.0f, score));
            }

            std::vector<float> scores;
            for (size_t i = 0; i < num_elements; ++i)
            {
                scores.push_back(std::sqrt(cls_scores[i] * iou_scores[i]));
                // if (cls_scores[i] * iou_scores[i] > 0.8)
                //     printf("cls_scores[i] * iou_scores[i] = %f\n", cls_scores[i] * iou_scores[i]);
            }

            float scale_w = static_cast<float>(input_size.first);
            float scale_h = static_cast<float>(input_size.second);

            // Calculate bboxes
            std::vector<float> bboxes(num_elements); // To store all bounding box coordinates
            std::vector<std::vector<float>> dets(num_elements, std::vector<float>(9));

            for (size_t i = 0; i < num_elements; ++i)
            {
                size_t loc_idx = i * 14;

                for (int j = 0; j < 4; ++j)
                {
                    float dx = loc[loc_idx + 4 + j * 2] * variance[0] * priors[i][2];
                    float dy = loc[loc_idx + 5 + j * 2] * variance[0] * priors[i][3];

                    float x = (priors[i][0] + dx) * scale_w;
                    float y = (priors[i][1] + dy) * scale_h;

                    dets[i][j * 2] = x;
                    dets[i][j * 2 + 1] = y;
                }
                dets[i][8] = scores[i];
            }

            return dets;
        }
    };

    class LPRcrnnPrePost
    {
    private:
        int LPR_input_image_width;
        int LPR_input_image_height;
        cv::Size maxTextSize;

        std::string plate_chr = "##########################################0123456789ABCDEFGHJKLMNPQRSTUVWXYZ##";

    public:
        LPRcrnnPrePost() {};
        ~LPRcrnnPrePost() {};

        void setInputImageSize(int img_width, int img_height)
        {
            LPR_input_image_width = img_width;
            LPR_input_image_height = img_height;
        }
        void preprocessing(void *rgb_buf, int width, int height, std::vector<float *> in_bufs)
        {
            const float mean_value = 0.588;
            const float std_value = 0.193;
            cv::Mat orig_mat(height, width, CV_8UC3, rgb_buf);
            cv::Mat resized_mat, normalized_img;

            cv::resize(orig_mat, resized_mat, cv::Size(LPR_INPUT_TENSOR_WIDTH, LPR_INPUT_TENSOR_HEIGHT), 0, 0, cv::INTER_LINEAR);
            resized_mat.convertTo(resized_mat, CV_32F, 1.0 / 255.0, 0 - mean_value);

            float *prelayer_processed_input = in_bufs[0];
            float *ifmap_rgb = (float *)resized_mat.data;
            for (int i = 0; i < (LPR_INPUT_TENSOR_WIDTH * LPR_INPUT_TENSOR_HEIGHT * LPR_INPUT_TENSOR_CHANNEL); i++)
                prelayer_processed_input[i] = (((float)ifmap_rgb[i]) / std_value);
        }

        void postprocessing(cv::Mat &image, std::vector<float *> out_bufs)
        {
            float *lp_chr = out_bufs[0];
            float *lp_color = out_bufs[1];
            std::vector<std::string> plate_color_list = {"black", "blue", "green", "white", "yellow"};

            int color_index = find_max_idx(lp_color, 0, 5);

            // Find the index of the maximum value along the last axis
            std::vector<int> char_idxes;
            for (int i = 0; i < 21; i++)
            {
                auto chr_idx = find_max_idx(lp_chr, i * 78, 78);
                chr_idx -= i * 78;
                char_idxes.push_back(chr_idx);
            }

            // Rest of the code similar to the previous response
            int pre = 0;
            std::vector<int> newPreds;

            for (size_t i = 0; i < char_idxes.size(); ++i)
            {
                if (char_idxes[i] != 0 && char_idxes[i] != pre)
                {
                    newPreds.push_back(char_idxes[i]);
                }
                pre = char_idxes[i];
            }

            std::string plate = "";
            for (int pred : newPreds)
            {
                plate += plate_chr[pred];
            }

            // Set the text parameters
            int fontFace = cv::FONT_HERSHEY_SIMPLEX;
            double fontScale = 0.7;        // Smaller scale factor for the text size
            int thickness = 2;             // Reduce thickness of the text lines
            cv::Scalar textColor(0, 0, 0); // Black color for the text (BGR format)

            // Define a smaller fixed text box size (width and height)
            int boxWidth = 150; // Smaller width for the text box
            int boxHeight = 40; // Smaller height for the text box

            // Calculate the size of the text to center it in the box
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(plate, fontFace, fontScale, thickness, &baseline);
            baseline += thickness;

            // Center the text within the box
            int textX = (boxWidth - textSize.width) / 2;
            int textY = (boxHeight + textSize.height) / 2;

            // Set the position for the box: upper right corner, with a 10-pixel margin from the edges
            cv::Point boxOrg(LPR_input_image_width - boxWidth - 10, 10);

            // Draw the fixed-size rectangle (text box) in the upper right corner
            cv::rectangle(image, boxOrg, boxOrg + cv::Point(boxWidth, boxHeight), cv::Scalar(255, 255, 255), cv::FILLED);

            // Draw the text inside the box, centered
            cv::putText(image, plate, boxOrg + cv::Point(textX, textY), fontFace, fontScale, textColor, thickness, cv::LINE_AA);
        }

        int find_max_idx(float *arr, int start, int size)
        {
            float max_value = arr[start]; // Initialize with the first element
            int max_index = start;

            for (int i = start + 1; i < start + size; ++i)
            {
                if (arr[i] > max_value)
                {
                    max_value = arr[i];
                    max_index = i;
                }
            }
            return max_index;
        }
    };

    // Model parameters
    int model_input_width;  // width of model input image
    int model_input_height; // height of model input image
    int input_image_width;  // width of input image
    int input_image_height; // height of input image
    uint32_t model_input_channel = 0;
    uint32_t model_output_height = 0;
    uint32_t model_output_width = 0;
    uint32_t model_output_channel = 0;

    int img_rows = 0;
    int img_cols = 0;

    std::condition_variable f_cond_recog_in, f_cond_recog_out, f_cond_detect_in, f_cond_detect_out;
    std::mutex f_mutex_recog_img, f_mutex_img, f_mutex_tmp_img, f_mutex_crop_img;

    // MX::Runtime::MxAccl *accl = NULL;
    SolutionManagerDualModel *solution_manager = NULL;
    LPDYunetPrePost *yunet_handle = nullptr;
    LPRcrnnPrePost *lprcrnn_handle = nullptr;
    std::vector<std::vector<float>> lpd_result;

    // Application Variables
    std::deque<cv::Mat> frames_queue; // Queue for frames
    std::mutex frame_queue_mutex;     // Mutex to control access to the queue
    int num_frames = 0;
    int frame_count = 0;
    float fps_number = .0; // FPS counter
    std::chrono::milliseconds start_ms;
    cv::VideoCapture vcap; // Video capture object
    std::vector<size_t> in_tensor_sizes;
    std::vector<size_t> out_tensor_sizes;
    MX::Types::MxModelInfo model_info; // Model info structure
    float *mxa_output;                 // Buffer for the output of the accelerator
    MxQt *gui_; // GUI for display

    void Stop()
    {
        // free input data buffers/output data buffers
        free_memory_model_input_output();

        // delete solution manager
        delete solution_manager;

        // delete handles
        delete yunet_handle;
        delete lprcrnn_handle;
    }

    void allocate_memory_model_input_output(
        MX::Types::MxModelInfo &model_info_0, MX::Types::MxModelInfo &model_info_1,
        MX::Types::MxModelInfo &post_model_info, int stream_buffer_size)
    {
        for (int i = 0; i < stream_buffer_size; i++)
        {
            // Model 0
            // Allocate memory for input feature maps
            std::vector<float *> input_data_0;
            for (int i = 0; i < model_info_0.num_in_featuremaps; ++i)
            {
                float *pData = new float[model_info_0.in_featuremap_sizes[i]];
                input_data_0.push_back(pData);
            }
            solution_manager->queue_input_data_0.push(input_data_0);

            // Allocate memory for output feature maps
            std::vector<float *> output_data_0;
            for (int i = 0; i < post_model_info.num_out_featuremaps; ++i)
            {
                float *pData = new float[post_model_info.out_featuremap_sizes[i]];
                output_data_0.push_back(pData);
            }
            solution_manager->queue_output_data_0.push(output_data_0);

            // Allocate memory for cropped image
            cv::Mat *img = new cv::Mat(input_image_width, input_image_height, CV_8UC3);
            solution_manager->queue_img_buffer.push(img);

            cv::Mat *img_cropped_buf = new cv::Mat();
            solution_manager->queue_crop_img_buffer.push(img_cropped_buf);

            // Model 1
            // Allocate memory for input feature maps
            std::vector<float *> input_data_1;
            for (int i = 0; i < model_info_1.num_in_featuremaps; ++i)
            {
                float *pData = new float[model_info_1.in_featuremap_sizes[i]];
                input_data_1.push_back(pData);
            }
            solution_manager->queue_input_data_1.push(input_data_1);

            // Allocate memory for output feature maps
            std::vector<float *> output_data_1;
            for (int i = 0; i < post_model_info.num_out_featuremaps; ++i)
            {
                float *pData = new float[post_model_info.out_featuremap_sizes[i]];
                output_data_1.push_back(pData);
            }
            solution_manager->queue_output_data_1.push(output_data_1);
        }
    }

    void free_memory_model_input_output()
    {
        // free input data buffers
        while (solution_manager->queue_input_data_0.size())
        {
            std::vector<float *> input_data = solution_manager->queue_input_data_0.pop();
            for (int i = 0; i < (int)input_data.size(); i++)
            {
                delete input_data.at(i);
            }
            input_data.clear();
        }

        // Free output data buffers
        while (solution_manager->queue_output_data_0.size())
        {
            std::vector<float *> output_data = solution_manager->queue_output_data_0.pop();
            for (int i = 0; i < (int)output_data.size(); i++)
            {
                delete output_data.at(i);
            }
            output_data.clear();
        }

        // Clear the queue
        for (int i = 0; i < STREAM_ALLOCATE_BUFFER; ++i)
        {
            while (solution_manager->queue_tmp_img.size())
            {
                cv::Mat *mat_ptr = solution_manager->queue_tmp_img.pop();
                if (mat_ptr)
                    delete mat_ptr;
            }
            while (solution_manager->queue_crop_img.size())
            {
                cv::Mat *mat_ptr = solution_manager->queue_crop_img.pop();
                if (mat_ptr)
                    delete mat_ptr;
            }
            while (solution_manager->queue_recog_img.size())
            {
                cv::Mat *mat_ptr = solution_manager->queue_recog_img.pop();
                if (mat_ptr)
                    delete mat_ptr;
            }
            while (solution_manager->queue_img_buffer.size())   
            {
                cv::Mat *mat_ptr = solution_manager->queue_img_buffer.pop();
                if (mat_ptr)
                    delete mat_ptr;
            }
            while (solution_manager->queue_crop_img_buffer.size())   
            {
                cv::Mat *mat_ptr = solution_manager->queue_crop_img_buffer.pop();
                if (mat_ptr)
                    delete mat_ptr;
            }
            
            
        }
    }

    // MX callbacks
    bool incallback_lp_detect(vector<const MX::Types::FeatureMap<float> *> dst, int streamLabel)
    {
        if (runflag.load())
        {
            cv::Mat inframe;
            cv::Mat rgbImage;
            bool got_frame = vcap.read(inframe);

            if (!got_frame)
            {
                std::cout << "No frame \n\n\n";
                vcap.release();
                
                return false; // return false if frame retrieval fails/stream is done sending input
            }
            cv::cvtColor(inframe, rgbImage, cv::COLOR_BGR2RGB);
            // Set input data to be sent to accelarator
            std::unique_lock<std::mutex> lock(f_mutex_img);

            // Wait until the queue is not empty or runflag.load() becomes false
            f_cond_detect_in.wait(lock, [&]()
                                  { return solution_manager->queue_input_data_0.size() > 0 &&
                                               solution_manager->queue_img_buffer.size() > 0 ||
                                           !runflag.load(); });

            std::vector<float *> input_data = solution_manager->queue_input_data_0.pop();
            cv::Mat *img = solution_manager->queue_img_buffer.pop();
            lock.unlock();
            rgbImage.copyTo(*img);

            // preprocess the image with given width and height
            yunet_handle->preprocessing(rgbImage.data, rgbImage.size().width, rgbImage.size().height, input_data);
            std::lock_guard<std::mutex> lock_tmp(f_mutex_tmp_img);
            solution_manager->queue_tmp_img.push(img);

            // Set preprocessed input data to be sent to accelerator
            for (int in_idx = 0; in_idx < dst.size(); ++in_idx)
            {
                dst[in_idx]->set_data(input_data[in_idx], false);
            }
            f_cond_detect_out.notify_one();

            solution_manager->queue_input_data_0.push(input_data);

            return true;
        }
        else
        {
            return false;
        }
    }

    bool outcallback_lp_detect(vector<const MX::Types::FeatureMap<float> *> src, int streamLabel)
    {
        if (runflag.load())
        {
            std::unique_lock<std::mutex> lock(f_mutex_tmp_img);

            // Wait until the queue is not empty or runflag.load() becomes false
            f_cond_detect_out.wait(lock, [&]()
                                { return solution_manager->queue_tmp_img.size() > 0 &&
                                        solution_manager->queue_output_data_0.size() > 0 ||
                                    !runflag.load(); });

            std::vector<float *> output_data = solution_manager->queue_output_data_0.pop();
            cv::Mat *img = solution_manager->queue_tmp_img.pop();
            lock.unlock();

            for (int out_idx = 0; out_idx < src.size(); ++out_idx)
            {
                src[out_idx]->get_data(output_data[out_idx], false);
            }

            // postprocess the inference result
            lpd_result = yunet_handle->postprocessing(output_data);

            if (lpd_result.size() == 0)
            {

                // using mx QT util to update the display frame
                gui_->screens[0]->SetDisplayFrame(streamLabel, img, fps_number);

                solution_manager->queue_img_buffer.push(img);
                solution_manager->queue_output_data_0.push(output_data);

                // Calulate FPS once every AVG_FPS_CALC_FRAME_COUNT frames
                frame_count++;
                if (frame_count == 1)
                {
                    start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
                }
                else if (frame_count % AVG_FPS_CALC_FRAME_COUNT == 0)
                {
                    std::chrono::milliseconds duration =
                        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()) - start_ms;
                    fps_number = (float)AVG_FPS_CALC_FRAME_COUNT * 1000 / (float)(duration.count());
                    frame_count = 0;
                }

                // clean up
                lpd_result.clear();

                return true;
            }

            while (solution_manager->queue_crop_img_buffer.size() == 0)
            {
                if (!runflag.load())
                    return false;
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }

            // draw bounding box on the original image
            cv::Mat *img_cropped_buf = solution_manager->queue_crop_img_buffer.pop();
            yunet_handle->draw_bounding_box(lpd_result, *img, *img_cropped_buf);
            solution_manager->queue_img_buffer.push(img);

            std::lock_guard<std::mutex> lock_crop(f_mutex_crop_img);
            solution_manager->queue_crop_img.push(img_cropped_buf);

            std::lock_guard<std::mutex> lock_recog(f_mutex_recog_img);
            solution_manager->queue_recog_img.push(img);

            f_cond_recog_in.notify_one();

            // clean up
            lpd_result.clear();

            solution_manager->queue_output_data_0.push(output_data);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool incallback_lp_recog(vector<const MX::Types::FeatureMap<float> *> dst, int streamLabel)
    {
        if (runflag.load())
        {
            // Create a unique lock on the mutex
            std::unique_lock<std::mutex> lock{f_mutex_crop_img};

            // Wait until the queue is not empty or runflag.load() becomes false
            f_cond_recog_in.wait(lock, [&]()
                                 { return (solution_manager->queue_crop_img.size() > 0 &&
                                           solution_manager->queue_input_data_1.size() > 0) ||
                                          !runflag.load(); });

            std::vector<float *> input_data = solution_manager->queue_input_data_1.pop();
            cv::Mat *cropped_image = solution_manager->queue_crop_img.pop();
            lock.unlock();

            lprcrnn_handle->preprocessing(cropped_image->data, cropped_image->size().width, cropped_image->size().height, input_data);

            // Set preprocessed input data to be sent to accelarator
            for (int in_idx = 0; in_idx < solution_manager->model_info_1.num_in_featuremaps; ++in_idx)
            {
                dst[in_idx]->set_data(input_data[in_idx], false);
            }

            solution_manager->queue_input_data_1.push(input_data);

            cropped_image->setTo(cv::Scalar(0, 0, 0));
            solution_manager->queue_crop_img_buffer.push(cropped_image);

            return true;
        }
        else
        {
            return false;
        }
    }

    bool outcallback_lp_recog(vector<const MX::Types::FeatureMap<float> *> src, int streamLabel)
    {
        if (runflag.load())
        {
            std::unique_lock<std::mutex> lock(f_mutex_recog_img);

            // Wait until the queue is not empty or runflag.load() becomes false
            f_cond_recog_out.wait(lock, [&]()
                                  { return !solution_manager->queue_recog_img.size() == 0 &&
                                               !solution_manager->queue_output_data_1.size() == 0 ||
                                           !runflag.load(); });

            std::vector<float *> output_data = solution_manager->queue_output_data_1.pop();
            cv::Mat *recog_img = solution_manager->queue_recog_img.pop();
            lock.unlock();

            for (int out_idx = 0; out_idx < solution_manager->model_info_1.num_out_featuremaps; ++out_idx)
            {
                src[out_idx]->get_data(output_data[out_idx], true);
            }

            // postprocess the inference result
            lprcrnn_handle->postprocessing(*recog_img, output_data);

            gui_->screens[0]->SetDisplayFrame(streamLabel, recog_img, fps_number);

            // Calulate FPS once every AVG_FPS_CALC_FRAME_COUNT frames
            frame_count++;
            if (frame_count == 1)
            {
                start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
            }
            else if (frame_count % AVG_FPS_CALC_FRAME_COUNT == 0)
            {
                std::chrono::milliseconds duration =
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()) - start_ms;
                fps_number = (float)AVG_FPS_CALC_FRAME_COUNT * 1000 / (float)(duration.count());
                frame_count = 0;
            }

            solution_manager->queue_output_data_1.push(output_data);

            return true;
        }
        else
        {
            return false;
        }
    }

public:
    ALPR(MX::Runtime::MxAccl *accl, std::string video_src, MxQt *gui, int index)
    {
        // Assigning gui variable to class specific variable
        gui_ = gui;
        // If the input is a camera, try to use optimal settings
        if (video_src.substr(0, 3) == "cam")
        {
            int device = std::stoi(video_src.substr(4));
#ifdef __linux__
            if (!openCamera(vcap, device, cv::CAP_V4L2))
            {
                throw(std::runtime_error("Failed to open: " + video_src));
            }

#elif defined(_WIN32)
            if (!openCamera(vcap, device, cv::CAP_ANY))
            {
                throw(std::runtime_error("Failed to open: " + video_src));
            }
#endif
        }
        else if (video_src.substr(0, 3) == "vid")
        {
            std::cout << "Video source given = " << video_src.substr(4) << "\n\n";
            vcap.open(video_src.substr(4), cv::CAP_ANY);
        }
        else
        {
            throw(std::runtime_error("Given video src: " + video_src + " is invalid" +
                                     "\n\n\tUse ./objectDetection cam:<camera index>,vid:<path to video file>,cam:<camera index>,vid:<path to video file>\n\n"));
        }
        if (!vcap.isOpened())
        {
            std::cout << "videocapture for " << video_src << " is NOT opened, \n try giving full absolete paths for video files and correct camera index for cmameras \n";
            runflag.store(false);
        }

        // Getting input image dimensions
        input_image_width = static_cast<int>(vcap.get(cv::CAP_PROP_FRAME_WIDTH));
        input_image_height = static_cast<int>(vcap.get(cv::CAP_PROP_FRAME_HEIGHT));

        solution_manager = new SolutionManagerDualModel();

        // allocate input & output buffers
        solution_manager->model_info_0 = accl->get_model_info(0);
        solution_manager->post_model_info = accl->get_post_model_info(0);
        solution_manager->model_info_1 = accl->get_model_info(1);

        allocate_memory_model_input_output(solution_manager->model_info_0, solution_manager->model_info_1, solution_manager->post_model_info, 5);

        accl->connect_stream(
            std::bind(&ALPR::incallback_lp_detect, this, std::placeholders::_1, std::placeholders::_2),
            std::bind(&ALPR::outcallback_lp_detect, this, std::placeholders::_1, std::placeholders::_2),
            index /*unique stream ID*/,
            0 /*Model ID */);

        accl->connect_stream(
            std::bind(&ALPR::incallback_lp_recog, this, std::placeholders::_1, std::placeholders::_2),
            std::bind(&ALPR::outcallback_lp_recog, this, std::placeholders::_1, std::placeholders::_2),
            index /*unique stream ID*/,
            1 /*Model ID */);

        // Init handles
        yunet_handle = new LPDYunetPrePost();
        lprcrnn_handle = new LPRcrnnPrePost();

        // setInputImageSize
        lprcrnn_handle->setInputImageSize(input_image_width, input_image_height);

        // Starts the callbacks when the call is started
        runflag.store(true);
    }

    ~ALPR()
    {
        Stop();
    }
};

int main(int argc, char *argv[])
{
    signal(SIGINT, signal_handler); // Set up signal handler
    vector<string> video_src_list;

    std::string video_str = "cam:0";

    // Iterate through the arguments
    for (int i = 1; i < argc; i++)
    {

        std::string arg = argv[i];
        
        // Handle --video_paths
        if (arg == "--video_paths")
        {
            if (i + 1 < argc && argv[i + 1][0] != '-')
            { // Ensure there's a next argument and it is not another option
                video_str = argv[++i];
                size_t pos = 0;
                std::string token;
                std::string delimiter = ",";
                while ((pos = video_str.find(delimiter)) != std::string::npos)
                {
                    token = video_str.substr(0, pos);
                    video_src_list.push_back(token);
                    video_str.erase(0, pos + delimiter.length());
                }
                video_src_list.push_back(video_str);
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

    // if video_paths arg isn't passed - use default video string.
    if (video_src_list.size() == 0)
    {
        video_src_list.push_back(video_str);
    }

    // Create the Accl object and load the DFP model
    MX::Runtime::MxAccl *accl = new MX::Runtime::MxAccl();
    accl->connect_dfp(model_path.c_str());

    // Connect the post-processing model
    accl->connect_post_model(postprocessing_model_path);

    // Creating GuiView for display
    MxQt gui(argc, argv);
    if (video_src_list.size() == 1)
        gui.screens[0]->SetSquareLayout(1, false); // Single stream layout
    else
        gui.screens[0]->SetSquareLayout(static_cast<int>(video_src_list.size())); // Multi-stream layout

    // Creating ALPR objects for each video stream
    std::vector<ALPR *> alpr_objs;
    for (int i = 0; i < video_src_list.size(); ++i)
    {
        ALPR *obj = new ALPR(accl, video_src_list[i], &gui, i);
        alpr_objs.push_back(obj);
    }

    // Run the accelerator and wait
    accl->start();
    gui.Run(); // Wait until the exit button is pressed in the Qt window
    accl->stop();

    // Cleanup
    delete accl;
    for (int i = 0; i < video_src_list.size(); ++i)
    {
        delete alpr_objs[i];
    }
}
