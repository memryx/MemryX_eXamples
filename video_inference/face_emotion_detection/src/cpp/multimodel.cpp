#include <iostream>
#include <thread>
#include <signal.h>

#include <opencv2/opencv.hpp>    /* imshow */
#include <opencv2/imgproc.hpp>   /* cvtcolor */
#include <opencv2/imgcodecs.hpp> /* imwrite */

#include "memx/accl/MxAccl.h"

#include <torch/torch.h>

#include "model.h"

MPFaceDetector model;

namespace fs = std::filesystem;

bool use_cam = true;
bool use_img = false;
std::atomic_bool runflag;

// video capture to read cmaera or file
cv::VideoCapture vcap;

double origHeight = 0.0;
double origWidth = 0.0;

const int AVG_FPS_CALC_FRAME_COUNT = 50;

// video file
fs::path videoPath; 
fs::path imagePath;

std::string emojiDir =  "../emojis";

// model file 
fs::path modelPath = "models/models.dfp";

cv::Mat image = cv::imread(imagePath);

//model info
MX::Types::MxModelInfo model_info;
MX::Types::MxModelInfo model_info_emotion;

//Queues to add input frames
std::deque<cv::Mat> frames_queue;
std::mutex frameQueue_Lock;

std::deque<cv::Mat> frames_queue_face;
std::mutex frameQueue_Lock_face;
std::condition_variable f_cond;

std::deque<cv::Mat> frames_queue_oface;
std::mutex frameQueue_Lock_oface;

std::deque<cv::Mat> frames_queue_emotion;
std::mutex frameQueue_Lock_emotion;

//Queus to add output from mxa
std::deque<std::vector<float*>> ofmap_queue;
std::mutex ofmap_queue_lock;

std::deque<std::vector<float*>> ofmap_queue_emotion;
std::mutex ofmap_queue_lock_emotion;

int model0_input_width = 128;
int model0_input_height = 128;

int model1_input_width = 224;
int model1_input_height = 224;

int frame_count = 0;
float fps_number =.0;
char fps_text[64] = "FPS = ";
std::chrono::milliseconds start_ms;

std::map<int, cv::Mat> idxToEmoji;

std::deque<int> emotion_queue;
std::map<int, int> emotion_ctr;
const int emotion_duration = 7;

bool window_created = false; // To make sure to open the window only once

//signal handler
void signalHandler(int pSignal){
    runflag.store(false);
}

void print_model_info_face(){

    std::cout << "\n******** Model Index : " << model_info.model_index << " ********\n";
    std::cout << "\nNum of in featuremaps : " << model_info.num_in_featuremaps << "\n";
    
    std::cout << "\nIn featureMap Shapes \n";
    for(int i = 0 ; i<model_info.num_in_featuremaps ; ++i){
        std::cout << "Shape of featureMap : " << i+1 << "\n";
        std::cout << "Layer Name : " << model_info.input_layer_names[i] << "\n";
        std::cout << "H = " << model_info.in_featuremap_shapes[i][0] << "\n";
        std::cout << "W = " << model_info.in_featuremap_shapes[i][1] << "\n";
        std::cout << "Z = " << model_info.in_featuremap_shapes[i][2] << "\n";
        std::cout << "C = " << model_info.in_featuremap_shapes[i][3] << "\n";
    }

    std::cout << "\n\nNum of out featuremaps : " << model_info.num_out_featuremaps << "\n";
    std::cout << "\nOut featureMap Shapes \n";
    for(int i = 0; i<model_info.num_out_featuremaps ; ++i){
        std::cout << "Shape of featureMap : " << i+1 << "\n";
        // std::cout << "Layer Name : " << model_info.output_layer_names[i] << "\n";
        std::cout << "H = " << model_info.out_featuremap_shapes[i][0] << "\n";
        std::cout << "W = " << model_info.out_featuremap_shapes[i][1] << "\n";
        std::cout << "Z = " << model_info.out_featuremap_shapes[i][2] << "\n";
        std::cout << "C = " << model_info.out_featuremap_shapes[i][3] << "\n";
    }

    std::cout << "Done printing model info \n";
}

void print_model_info_emotion(){

    std::cout << "\n******** Model Index : " << model_info_emotion.model_index << " ********\n";
    std::cout << "\nNum of in featuremaps : " << model_info_emotion.num_in_featuremaps << "\n";
    
    std::cout << "\nIn featureMap Shapes \n";
    for(int i = 0 ; i<model_info_emotion.num_in_featuremaps ; ++i){
        std::cout << "Shape of featureMap : " << i+1 << "\n";
        std::cout << "Layer Name : " << model_info_emotion.input_layer_names[i] << "\n";
        std::cout << "H = " << model_info_emotion.in_featuremap_shapes[i][0] << "\n";
        std::cout << "W = " << model_info_emotion.in_featuremap_shapes[i][1] << "\n";
        std::cout << "Z = " << model_info_emotion.in_featuremap_shapes[i][2] << "\n";
        std::cout << "C = " << model_info_emotion.in_featuremap_shapes[i][3] << "\n";
    }

    std::cout << "\n\nNum of out featuremaps : " << model_info_emotion.num_out_featuremaps << "\n";
    std::cout << "\nOut featureMap Shapes \n";
    for(int i = 0; i<model_info_emotion.num_out_featuremaps ; ++i){
        std::cout << "Shape of featureMap : " << i+1 << "\n";
        // std::cout << "Layer Name : " << model_info.output_layer_names[i] << "\n";
        std::cout << "H = " << model_info_emotion.out_featuremap_shapes[i][0] << "\n";
        std::cout << "W = " << model_info_emotion.out_featuremap_shapes[i][1] << "\n";
        std::cout << "Z = " << model_info_emotion.out_featuremap_shapes[i][2] << "\n";
        std::cout << "C = " << model_info_emotion.out_featuremap_shapes[i][3] << "\n";
    }

    std::cout << "Done printing model info \n";
}

std::vector<int> _get_box(const torch::Tensor& det, const cv::Mat& img) {
    int origWidth = img.cols; 
    int origHeight = img.rows;

    int ymin = static_cast<int>(det[0].item<float>() * origHeight);
    int xmin = static_cast<int>(det[1].item<float>() * origWidth);
    int ymax = static_cast<int>(det[2].item<float>() * origHeight);
    int xmax = static_cast<int>(det[3].item<float>() * origWidth);
    return {xmin, ymin, xmax, ymax};
}

// // Function to crop the first detected face from the image
cv::Mat cropFirstDetectedFace(cv::Mat& img, const torch::Tensor& dets) {
    if (dets.size(0) > 0) {
        torch::Tensor dets_cpu = dets.to(torch::kCPU);
        auto box = _get_box(dets_cpu[0], img);

        int xmin = std::max(box[0], 0);
        int ymin = std::max(box[1], 0);
        int xmax = std::min(box[2], img.cols);
        int ymax = std::min(box[3], img.rows);

        if (xmax > xmin && ymax > ymin) {
            cv::Rect roi(xmin, ymin, xmax - xmin, ymax - ymin);
            return img(roi).clone();
        }
    }
    // Return a blank frame of the same size and type as the input image
    return cv::Mat::zeros(img.size(), img.type());
}

std::vector<int> _get_box_orig(torch::Tensor det, cv::Mat& img) {

    int origWidth = img.cols; 
    int origHeight = img.rows;

    int ymin = static_cast<int>(det[0].item<float>() * origHeight);
    int xmin = static_cast<int>(det[1].item<float>() * origWidth);
    int ymax = static_cast<int>(det[2].item<float>() * origHeight);
    int xmax = static_cast<int>(det[3].item<float>() * origWidth);
    return {xmin, ymin, xmax, ymax};
}

void draw(cv::Mat& img, torch::Tensor dets) {
    
    dets = dets.to(torch::kCPU);

    for (int i = 0; i < dets.size(0); ++i) {
        auto box = _get_box_orig(dets[i], img);
        cv::rectangle(img, cv::Point(box[0], box[1]), cv::Point(box[2], box[3]), cv::Scalar(255, 0, 0), 3);
    }
}

void preloadEmojis(const std::string& emojiDir, std::map<int, cv::Mat>& idxToEmoji) {
    // Define the mapping from indices to emoji filenames
    std::map<int, std::string> idxToFilename = {
        {0, "1F92C_color.png"},
        {1, "1F92E_color.png"},
        {2, "1F628_color.png"},
        {3, "1F604_color.png"},
        {4, "1F610_color.png"},
        {5, "1F622_color.png"},
        {6, "1F62F_color.png"},
        {7, "1F50D_color.png"}
    };

    // Iterate over the mapping, load each emoji, and store it in idxToEmoji
    for (const auto& pair : idxToFilename) {
        std::string fullPath = emojiDir + "/" + pair.second; // Construct the full path
        cv::Mat emoji = cv::imread(fullPath, cv::IMREAD_UNCHANGED); // Load the emoji with alpha channel

        if (!emoji.empty()) {
            cv::Mat resizedEmoji;
            cv::resize(emoji, resizedEmoji, cv::Size(128, 128), 0, 0, cv::INTER_CUBIC); // Resize the emoji
            idxToEmoji[pair.first] = resizedEmoji; // Store the resized emoji in the map
        } else {
            std::cerr << "Failed to load emoji: " << fullPath << std::endl;
        }
    }
}

int smoothEmotion(int emotion_idx) {
    if (emotion_queue.size() == emotion_duration) {
        int oldest = emotion_queue.front();
        emotion_queue.pop_front();
        emotion_ctr[oldest]--;
        if (emotion_ctr[oldest] == 0) {
            emotion_ctr.erase(oldest);
        }
    }
    emotion_queue.push_back(emotion_idx);
    emotion_ctr[emotion_idx]++;

    int argmax = -1;
    int max_count = 0;
    for (auto& pair : emotion_ctr) {
        if (pair.second > max_count) {
            max_count = pair.second;
            argmax = pair.first;
        }
    }
    return argmax;
}

void drawEmoji(cv::Mat& img, int idx) {
    if (idxToEmoji.find(idx) == idxToEmoji.end()) {
        std::cout << "drawwwwww" << "\n";
        return;
    }

    cv::Mat emoji = idxToEmoji[idx];
    if (emoji.channels() != 4) {
        std::cerr << "Emoji does not have an alpha channel" << std::endl;
        return;
    }

    std::vector<cv::Mat> emojiChannels(4);
    cv::split(emoji, emojiChannels);
    cv::Mat alphaMask = emojiChannels[3]; // Alpha channel
    cv::Mat emojiBGR;
    cv::merge(std::vector<cv::Mat>{emojiChannels[0], emojiChannels[1], emojiChannels[2]}, emojiBGR);

    int x = 25; // X position to draw the emoji
    int y = 25; // Y position to draw the emoji

    // Resize emoji if larger than the image
    if (emojiBGR.cols + x > img.cols || emojiBGR.rows + y > img.rows) {
        int newWidth = std::min(emojiBGR.cols, img.cols - x);
        int newHeight = std::min(emojiBGR.rows, img.rows - y);
        cv::resize(emojiBGR, emojiBGR, cv::Size(newWidth, newHeight));
        cv::resize(alphaMask, alphaMask, cv::Size(newWidth, newHeight));
    }

    // Position the emoji on the image
    cv::Rect roiRect(x, y, emojiBGR.cols, emojiBGR.rows);
    cv::Mat roi = img(roiRect);

    // Prepare the emoji and the mask for blending
    cv::Mat imgFloat, emojiFloat, maskFloat;
    roi.convertTo(imgFloat, CV_32F);
    emojiBGR.convertTo(emojiFloat, CV_32F);
    alphaMask.convertTo(maskFloat, CV_32F, 1.0 / 255); // Normalize mask to [0, 1]

    // Prepare the mask for BGR channels
    std::vector<cv::Mat> maskChannels = {maskFloat, maskFloat, maskFloat};
    cv::Mat maskBGR;
    cv::merge(maskChannels, maskBGR);

    // Blend the emoji onto the image
    cv::Mat blendedPart = (emojiFloat.mul(maskBGR) + imgFloat.mul(cv::Scalar(1.0, 1.0, 1.0) - maskBGR));
    blendedPart.convertTo(roi, img.type());

    return;
}

int getEmotionIndex(const std::vector<float*>& dets) {
    if (dets.empty() || dets[0] == nullptr) {
        std::cerr << "Detections are empty or null." << std::endl;
        return -1; // Indicate an error or unknown emotion
    }

    const int numEmotions = 7; 
    int maxIndex = 0;
    float maxScore = dets[0][0]; // Initialize with the first score

    for (int i = 1; i < numEmotions; ++i) {
        if (dets[0][i] > maxScore) {
            maxScore = dets[0][i];
            maxIndex = i;
        }
    }

    return maxIndex;
}

cv::Mat preprocess_face( cv::Mat& image ) {
    
    // Resize the image using cubic interpolation
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(model0_input_width, model0_input_height), 0, 0, cv::INTER_CUBIC);

    // Convert BGR image to RGB
    cv::Mat rgbImage;
    cv::cvtColor(resizedImage, rgbImage, cv::COLOR_BGR2RGB);

    // Convert image to float32
    cv::Mat floatImage;
    rgbImage.convertTo(floatImage, CV_32F);

    // Normalize pixel values to [-1, 1]
    cv::Mat normalizedImage = (floatImage / 127.5) - 1.0;

    return normalizedImage;
}

bool incallback_getframe_face(std::vector<const MX::Types::FeatureMap*> dst, int streamLabel){

    if(runflag.load()){
        bool got_frame = false;
        cv::Mat inframe;
        
        if(use_cam){
            got_frame = vcap.read(inframe);
        }

        else if(use_img){
            inframe = image; 
            if(!inframe.empty()){
                got_frame = true;
            }
            image.release();
        }

        else{
            got_frame = vcap.read(inframe);
        }

        if (!got_frame) {
            std::cout << "\n\n No frame - End of video/cam/img \n\n\n";
            runflag.store(false);
            return false;  // return false if frame retrieval fails
        }

        else{

            // Put the frame in the cap_queue to be overlayed later
            {
                std::unique_lock<std::mutex> flock(frameQueue_Lock);
                frames_queue.push_back(inframe);
            }

            // Preprocess frame
            cv::Mat preProcframe = preprocess_face(inframe);

            // Set preprocessed input data to be sent to accelarator
            dst[0]->set_data((float*)preProcframe.data);

            return true;
        }           
    }
    else{
        vcap.release();
        runflag.store(false);
        return false;
    }    
}

// Output callback function
bool outcallback_getmxaoutput_face(std::vector<const MX::Types::FeatureMap*> src, int streamLabel){

    std::vector<float*> ofmap;
    ofmap.reserve(src.size());
    for(int i = 0; i<model_info.num_out_featuremaps ; ++i){
        float * fmap = new float[model_info.out_featuremap_sizes[i]];
        src[i]->get_data(fmap);
        ofmap.push_back(fmap);
    }

    cv::Mat frame;
    cv::Mat inframe;

    {
        std::unique_lock<std::mutex> ilock(frameQueue_Lock);
        // pop from frame queue
        frame = frames_queue.front();
        frames_queue.pop_front();
    } // releases in frame queue lock

    torch::Tensor dets = model.postprocess(ofmap);

    draw(frame, dets);

    inframe = cropFirstDetectedFace(frame, dets);

    {
        std::unique_lock<std::mutex> flock(frameQueue_Lock_face);
        frames_queue_face.push_back(inframe);
        f_cond.notify_one();
    }

    {
        std::unique_lock<std::mutex> flock(frameQueue_Lock_oface);
        frames_queue_oface.push_back(frame);
    }

    for (auto& fmap : ofmap) {
        delete[] fmap;
        fmap = NULL;
    }
    
    return true;
}

cv::Mat preprocess_emotion(cv::Mat img, bool mirror) {

    int vertMargin = 20;
    int horzMargin = 150;
    
    if (mirror) {
        cv::flip(img, img, 1);
    }

    int adjustedVertMargin = std::min(vertMargin, img.rows / 4);
    int adjustedHorzMargin = std::min(horzMargin, img.cols / 4);

    // Check if there's enough space to crop
    if (img.rows <= 2 * adjustedVertMargin || img.cols <= 2 * adjustedHorzMargin) {
        std::cerr << "Adjusted margins still too large: Vertical margin = "
                  << adjustedVertMargin << ", Horizontal margin = " << adjustedHorzMargin << std::endl;
        // If dimensions are still too small, resize without cropping
    } else {
        // Crop the central part of the image
        cv::Rect roi(adjustedHorzMargin, adjustedVertMargin,
                     img.cols - 2 * adjustedHorzMargin,
                     img.rows - 2 * adjustedVertMargin);
        img = img(roi);
    }

    // Resize the image to 224x224
    cv::Mat resizedImg;
    cv::resize(img, resizedImg, cv::Size(224, 224), 0, 0, cv::INTER_CUBIC);

    // Convert image to float and subtract mean values
    cv::Mat floatImg;
    resizedImg.convertTo(floatImg, CV_32FC3);
    cv::subtract(floatImg, cv::Scalar(103.939, 116.779, 123.68), floatImg);

    return floatImg;
}

bool incallback_getframe_emotion(std::vector<const MX::Types::FeatureMap*> dst, int streamLabel){

   if(runflag.load()){

       cv::Mat inframe;
       {
           std::unique_lock<std::mutex> lock(frameQueue_Lock_face);
           auto now = std::chrono::steady_clock::now();
           if(!f_cond.wait_until(lock,now+1000ms, [](){ return !frames_queue_face.empty(); }))
           {
            runflag.store(false);
            return false;
           }
           // At this point, the lock is re-acquired after the wait
           inframe = frames_queue_face.front();
           frames_queue_face.pop_front();
       }
       // Preprocess frame
       cv::Mat preProcframe = preprocess_emotion(inframe, true);

       // Set preprocessed input data to be sent to accelarator
       dst[0]->set_data((float*)preProcframe.data);

       return true;
   }
   else{
       runflag.store(false);
       return false;
   }   
}

// Output callback function
bool outcallback_getmxaoutput_emotion(std::vector<const MX::Types::FeatureMap*> src, int streamLabel){

    std::vector<float*> ofmap_emotion;
    cv::Mat inframe;

    ofmap_emotion.reserve(src.size());

    for(int i=0; i<model_info_emotion.num_out_featuremaps ; ++i){
        float * fmap_emotion = new float[model_info_emotion.out_featuremap_sizes[i]];
        src[i]->get_data(fmap_emotion);
        ofmap_emotion.push_back(fmap_emotion);
    }

    {
        std::unique_lock<std::mutex> lock(frameQueue_Lock_oface);
        inframe = frames_queue_oface.front();
        frames_queue_oface.pop_front();
    }

    int emotionIdx = getEmotionIndex(ofmap_emotion); // Use the function to find the emotion index

    emotionIdx = smoothEmotion(emotionIdx);

    drawEmoji(inframe, emotionIdx);

    for (auto& fmap_emotion : ofmap_emotion) {
        delete[] fmap_emotion;
        fmap_emotion = NULL;
    }

    if(!window_created){

        cv::namedWindow("Face Detection", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
        cv::resizeWindow("Face Detection", cv::Size(640,480));
        cv::moveWindow("Face Detection", 0, 0);
        window_created=true;
    }

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

    sprintf(fps_text, "FPS = %.1f", fps_number);
    std::cout << "\r" << fps_text << "\t" << std::flush;

    cv::putText(inframe,fps_text,
        cv::Point2i(10, 30), // origin of text (bottom left of textbox)
        cv::FONT_ITALIC,
        0.8, // font scale
        cv::Scalar(255, 255, 0), // color (green)
        2 // thickness
    );

    // Display the image with detections
    cv::imshow("Face Detection", inframe);
    
    if(use_img){
        cv::waitKey(1000);
        runflag.store(false);
    }

    else{
        if (cv::waitKey(1) == 'q') {
            runflag.store(false);
        }
    }

    return true;
}

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

void run_inference(){

    runflag.store(true);
    
    if(use_cam){
        std::cout<<"use cam"<<"\n";

        #ifdef __linux__
            std::cout << "Running on Linux" << "\n";
            if (!openCamera(vcap, 0, cv::CAP_V4L2)) {
                return;
            }

        #elif defined(_WIN32)
            std::cout << "Running on Windows" << "\n";
            if (!openCamera(vcap, 0, cv::CAP_ANY)) {
                return;
            }
        #endif
    }

    else if(use_img){
        origWidth = image.cols; 
        origHeight = image.rows;
        if(image.empty()){
            std::cout << "Error loading the image\n";
            runflag.store(false);
        }
    }
    else {
        vcap.open(videoPath.c_str());
    }

    if(vcap.isOpened()){
        std::cout << "videocapture opened \n";

        origWidth = vcap.get(cv::CAP_PROP_FRAME_WIDTH); //get the width of frames of the video 
        origHeight = vcap.get(cv::CAP_PROP_FRAME_HEIGHT);

    }
    else if(use_img){
        std::cout << "image opened \n";
    }
    else{
        std::cout << "videocapture NOT opened \n";
        runflag.store(false);
    }

    if(runflag.load()){
    
        MX::Runtime::MxAccl accl{fs::path(modelPath)};

        model_info = accl.get_model_info(0);
        print_model_info_face();
        model0_input_height = model_info.in_featuremap_shapes[0][0];
        model0_input_width = model_info.in_featuremap_shapes[0][1];

        accl.connect_stream(&incallback_getframe_face, &outcallback_getmxaoutput_face, 0 /*unique stream ID*/, 0 /*Model ID */);   
        model_info_emotion = accl.get_model_info(1);
        print_model_info_emotion();
        model1_input_height = model_info_emotion.in_featuremap_shapes[0][0];
        model1_input_width = model_info_emotion.in_featuremap_shapes[0][1];

        accl.connect_stream(&incallback_getframe_emotion, &outcallback_getmxaoutput_emotion, 0 /*unique stream ID*/, 1 /*Model ID */);     
        
        std::cout << "Connected stream \n\n\n";

        accl.start();
        //accl.wait();
        while(runflag.load()){
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }  
        cv::destroyAllWindows();
        accl.stop();
        std::cout << "\n\rAccl stop called \n";  
    }
}

int main(int argc, char* argv[]){

    // Check if arguments are passed
    if (argc > 1) {

        preloadEmojis(emojiDir, idxToEmoji);

        for (int i = 1; i < argc; ++i) {
            std::string inputType(argv[i]);

            // Handle -d argument to set DFP path
            if (inputType == "-d" && i + 1 < argc) {
                modelPath = argv[++i];  // Set DFP path from next argument
            }
            // Handle --cam option
            else if (inputType == "--cam") {
                use_cam = true;
                runflag.store(true);
            }
            // Handle --video option, expects a path
            else if (inputType == "--video" && i + 1 < argc) {
                videoPath = argv[++i];  // Set video file path
                use_cam = false;
                runflag.store(true);
            }
            // Handle --img option, expects a path
            else if (inputType == "--img" && i + 1 < argc) {
                imagePath = argv[++i];  // Set image file path
                use_img = true;
                use_cam = false;
                runflag.store(true);
            }
            // Handle incorrect or missing arguments
            else {
                std::cout << "\n\nIncorrect or Missing Argument Passed \n";
                std::cout << "Usage: ./face_emotion_classification [--cam or --video <path> or --img <path>] [-d <dfp_path>]\n\n\n";
                runflag.store(false);
                return 1;
            }
        }
    }
    else {
        std::cout << "\n\nNo Arguments Passed \n";
        std::cout << "Usage: ./face_emotion_classification [--cam or --video <path> or --img <path>] [-d <dfp_path>]\n\n\n";
        runflag.store(false);
        return 1;
    }

    signal(SIGINT, signalHandler);

    if(runflag.load()){

        std::cout << "application start \n";
        std::cout << "model path = " << modelPath.c_str() << "\n";

        run_inference();
    }

    else{
        std::cout << "App exiting without execution \n\n\n";       
    }

    return 1;
}
