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

std::atomic_bool runflag;

//YoloV7 application specific parameters
fs::path model_path = "YOLO_v7_tiny_416_416_3_onnx.dfp";
fs::path postprocessing_model_path = "YOLO_v7_tiny_416_416_3_onnx_post.onnx";
#define AVG_FPS_CALC_FRAME_COUNT  50
#define FRAME_QUEUE_MAX_LENGTH     5
//signal handler
void signal_handler(int p_signal){
    runflag.store(false);
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


//Struct to hold detection outputs
struct detectedObj {
    int x1;
    int x2;
    int y1;
    int y2;
    int obj_id;
    float accuracy;

    detectedObj(int x1_, int x2_, int y1_, int y2_, int obj_id_, float accuracy_) {
       x1 = x1_;
       x2 = x2_;
       y1 = y1_;
       y2 = y2_;
       obj_id = obj_id_;
       accuracy = accuracy_;
   }
} ;

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

class YoloV7{
    private:
        // Model Params
        int model_input_width;//width of model input image
        int model_input_height;//height of model input image
        int input_image_width;//width of input image
        int input_image_height;//height of input image
        int num_boxes = 300;//Maximum number of boxes that can be output by the yolov7-tiny model
        float conf_thresh = 0.4;//Confidence threshold of the boxes
        std::vector<std::string> class_names = { //Class names list of COCO dataset
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

        //Application Variables
        std::deque<cv::Mat> frames_queue;
        std::mutex frame_queue_mutex;
        int num_frames=0;
        int frame_count = 0;
        float fps_number =.0;
        std::chrono::milliseconds start_ms;
        cv::VideoCapture vcap;
        bool src_is_cam = false;
        std::vector<size_t> in_tensor_sizes;
        std::vector<size_t> out_tensor_sizes;
        MX::Types::MxModelInfo model_info;
        float* mxa_output;
        cv::Mat displayImage;
        MxQt* gui_;

        cv::Mat preprocess( cv::Mat& image ) {

            cv::Mat resizedImage;
            cv::dnn::blobFromImage(image, resizedImage, 1.0, cv::Size(model_input_width, model_input_height), cv::Scalar(0, 0, 0), true, false);

            // Convert image to float32 and normalize
            cv::Mat floatImage;
            resizedImage.convertTo(floatImage, CV_32F, 1.0 / 255.0);

            return floatImage;
        }

        void draw_bounding_box(cv::Mat& image, std::vector<detectedObj>& detections_vector){
            for(int i=0;i<detections_vector.size();++i ) {
                detectedObj detected_object = detections_vector[i];
                cv::rectangle(image, cv::Point(detected_object.x1, detected_object.y1), cv::Point(detected_object.x2, detected_object.y2), cv::Scalar(0, 255, 0), 2);

                cv::putText(image, class_names.at( detected_object.obj_id ),
                            cv::Point(detected_object.x1, detected_object.y1 - 3), cv::FONT_ITALIC,
                            0.8, cv::Scalar(255, 255, 255), 2);

                cv::putText(image, std::to_string(detected_object.accuracy),
                            cv::Point(detected_object.x1, detected_object.y1+30), cv::FONT_ITALIC,
                            0.8, cv::Scalar(255, 255, 0), 2);
            }
        }

        std::vector<detectedObj> get_detections(float* output, int num_boxes){
            std::vector<detectedObj> detections;
            for (int i = 0; i < num_boxes; i++) {
                //Decoding model output
                float accuracy          = output[i * 7 + 6];
                if(accuracy<conf_thresh){
                    continue;
                }
                float x1                = output[i * 7 + 1];
                float y1                = output[i * 7 + 2];
                float x2                = output[i * 7 + 3];
                float y2                = output[i * 7 + 4];
                int classPrediction     = output[i * 7 + 5];

                // Coords should be scaled to the dispaly image. The coords from the model are relative to the model's input height and width.
                x1 = (x1  / model_input_width) * input_image_width ;
                x2 = (x2/ model_input_width) * input_image_width ;
                y1 = (y1 / model_input_height) * input_image_height ;
                y2 = (y2/ model_input_height) * input_image_height ;

                detectedObj obj( x1, x2, y1, y2, classPrediction, accuracy);

                detections.push_back( obj );
            }
            return detections;
        }

        bool incallback_getframe(std::vector<const MX::Types::FeatureMap*> dst, int streamLabel){

            if(runflag.load()){
                cv::Mat inframe;
                cv::Mat rgbImage;

                while(true){
                    bool got_frame = vcap.read(inframe);

                    if (!got_frame) {
                        std::cout << "No frame \n\n\n";
                        vcap.release();
                        return false;  // return false if frame retrieval fails/stream is done sending input
                    }
                    if(src_is_cam && (frames_queue.size() >= FRAME_QUEUE_MAX_LENGTH)){
                        // drop the frame and try again if we've hit the limit
                        continue;
                    }
                    else{
                        cv::cvtColor(inframe, rgbImage, cv::COLOR_BGR2RGB);
                        {
                            std::lock_guard<std::mutex> ilock(frame_queue_mutex);
                            frames_queue.push_back(rgbImage);
                        }
                    }
                    
                    // Preprocess frame
                    cv::Mat preProcframe = preprocess(rgbImage);
                    // Set preprocessed input data to be sent to accelarator
                    dst[0]->set_data((float*)preProcframe.data);

                    return true;
                }
            }
            else{
                vcap.release();
                return false;// Manually stopped the application so returning false to stop input.
            }    
        }

        // Input callback function to fetch frames and preprocess them
        bool outcallback_getmxaoutput(std::vector<const MX::Types::FeatureMap*> src, int streamLabel){
            
            //Ouput from the post-processing model is a vector of size 1
            //So copying only the first featuremap
            src[0]->get_data(mxa_output);
            // cv::Mat inImage;
            {
                std::lock_guard<std::mutex> ilock(frame_queue_mutex);
                // pop from frame queue
                displayImage = frames_queue.front();
                frames_queue.pop_front();
            }// releases in frame queue lock

            //Get the detections from model output
            std::vector<detectedObj> detected_objectVector = get_detections(mxa_output, num_boxes);
            
            // draw boundign boxes
            draw_bounding_box(displayImage, detected_objectVector );

            // using mx QT util to update the display frame
            gui_->screens[0]->SetDisplayFrame(streamLabel,&displayImage,fps_number);

            //Calulate FPS once every AVG_FPS_CALC_FRAME_COUNT frames     
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
            return true;
        }

    public:
        YoloV7(MX::Runtime::MxAccl* accl, std::string video_src, MxQt* gui, int index){
            //Assigning gui variable to class specifc variable
            gui_ = gui;
            // If the input is a camera, try to use optimal settings
            if(video_src.substr(0,3) == "cam"){
                int device = std::stoi(video_src.substr(4));
                src_is_cam = true;
                #ifdef __linux__
                    if (!openCamera(vcap, device, cv::CAP_V4L2)) {
                        throw(std::runtime_error("Failed to open: "+video_src));
                    }

                #elif defined(_WIN32)
                    if (!openCamera(vcap, device, cv::CAP_ANY)) {
                        throw(std::runtime_error("Failed to open: "+video_src));
                    }
                #endif
            }
            else if(video_src.substr(0,3) == "vid"){
                std::cout<<"Video source given = "<<video_src.substr(4) << "\n\n";
                vcap.open(video_src.substr(4), cv::CAP_ANY);
                src_is_cam = false;
            }
            else{
                throw(std::runtime_error("Given video src: "+video_src+" is invalid"+
                "\n\n\tUse ./objectDetection cam:<camera index>,vid:<path to video file>,cam:<camera index>,vid:<path to video file>\n\n"));
            }
            if(!vcap.isOpened()){
                std::cout << "videocapture for "<<video_src<<" is NOT opened, \n try giving full absolete paths for video files and correct camera index for cmameras \n";
                runflag.store(false);
            }

            // Getting input image dimensions
            input_image_width = static_cast<int>(vcap.get(cv::CAP_PROP_FRAME_WIDTH));
            input_image_height = static_cast<int>(vcap.get(cv::CAP_PROP_FRAME_HEIGHT));

            model_info = accl->get_model_info(0);//Getting model info of 0th model which is the only model in this DFP
            mxa_output= new float[num_boxes*7];//Creating the memory of output (max_boxes X num_box_parameters) 

            //Getting model input shapes and display size
            model_input_height = model_info.in_featuremap_shapes[0][0];
            model_input_width = model_info.in_featuremap_shapes[0][1];

            //Connecting the stream to the accl object. As the callback functions are defined as part of the class
            //YoloV7 we should bind them with the possible input parameters
            auto in_cb = std::bind(&YoloV7::incallback_getframe, this, std::placeholders::_1, std::placeholders::_2);
            auto out_cb = std::bind(&YoloV7::outcallback_getmxaoutput, this, std::placeholders::_1, std::placeholders::_2);
            accl->connect_stream(in_cb, out_cb, index/**Unique Stream Idx */, 0/**Model Idx */);

            //Starts the callbacks when the call is started
            runflag.store(true);
        }
        ~YoloV7(){
            delete[] mxa_output;
            mxa_output = NULL;
        }
};

int main(int argc, char* argv[]){
    signal(SIGINT, signal_handler);
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

    // Initialize the MemryX accelerator
    MX::Runtime::MxAccl accl{fs::path(model_path)};
    
    // Connecting the post-processing model obtained from the autocrop of neural compiler to get the final output.
    // The second parameter is required as the output shape of this particular post-processing model is variable
    // and accl requires to know maximum possible size of the output. In this case it is (max_possible_boxes * size_of_box = 300 *7= 2100).
    accl.connect_post_model(fs::path(postprocessing_model_path), 0, std::vector<size_t>{300*7});
    
    // Creating GuiView which is a memryx qt util for easy display
    MxQt gui(argc,argv);
    // Setting the layout of the display based on number of input streams. Full screen mode only when more than one stream
    if(video_src_list.size()==1)
        gui.screens[0]->SetSquareLayout(1,false);
    else
        gui.screens[0]->SetSquareLayout(static_cast<int>(video_src_list.size()));

    //Creating a YoloV7 object for each stream which also connects the corresponding stream to accl.
    std::vector<YoloV7*>yolo_objs;
    for(int i =0; i<video_src_list.size();++i){
        YoloV7* obj = new YoloV7(&accl,video_src_list[i],&gui,i);
        yolo_objs.push_back(obj);
    }

    //Run the accelerator and wait
    accl.start();
    gui.Run(); //This command waits for exit to be pressed in Qt window
    accl.stop();

    //Cleanup
    for(int i =0; i<video_src_list.size();++i ){
        delete yolo_objs[i];
    }
}