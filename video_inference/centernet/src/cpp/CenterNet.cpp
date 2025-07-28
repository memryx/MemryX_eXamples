#include "memx/accl/MxAccl.h"
#include <signal.h>
#include <iostream>
#include <opencv2/opencv.hpp>    /* imshow */
#include <opencv2/imgproc.hpp>   /* cvtcolor */
#include <opencv2/imgcodecs.hpp> /* imwrite */
#include <chrono>
#include <memx/mxutils/gui_view.h>

namespace fs = std::filesystem;

std::atomic_bool runflag;

#define AVG_FPS_CALC_FRAME_COUNT  50
#define FRAME_QUEUE_MAX_LENGTH     5

//CenterNet application specific Onnx model files
fs::path onnx_model_path = "models/centernet_onnx.dfp";
fs::path onnx_preprocessing_model_path = "models/centernet_pre.onnx";
fs::path onnx_postprocessing_model_path = "models/centernet_post.onnx";

//CenterNet application specific Tf model files
fs::path tf_model_path = "models/centernet_tf.dfp";
fs::path tf_preprocessing_model_path = "models/centernet_pre.pb";
fs::path tf_postprocessing_model_path = "models/centernet_post.pb";

//CenterNet application specific Tflite model files
fs::path tflite_model_path = "models/centernet_tflite.dfp";
fs::path tflite_preprocessing_model_path = "models/centernet_pre.tflite";
fs::path tflite_postprocessing_model_path = "models/centernet_post.tflite";

enum AppType{
    App_Onnx,
    App_Tflite,
    App_Tf    
};

//signal handler
void signal_handler(int p_signal){
    runflag.store(false);
}

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
            throw std::runtime_error("unknown input source passed");
        }
    }

    return true;
}

class CenterNet{
    private:
        MxQt* gui_;
        // Model Params
        int model_input_width;//width of model input image
        int model_input_height;//height of model input image
        int input_image_width;//width of input image
        int input_image_height;//height of input image
        int num_boxes;//Number of boxes that can be output by the CenterNet model
        float conf_thresh = 0.25;//Confidence threshold of the boxes
        std::vector<std::string> class_names = {
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

        //Applcation variables
        std::deque<cv::Mat> frames_queue;
        std::mutex frame_queue_mutex;
        int num_frames=0;
        int frame_count = 0;
        float fps_number =.0;
        std::chrono::milliseconds start_ms;
        cv::VideoCapture vcap;
        bool src_is_cam = false;
        MX::Types::MxModelInfo model_info;
        MX::Types::MxModelInfo post_model_info;
        cv::Mat displayImage;
        vector<float*> output;
        AppType type_; 
        struct output_map
        {
            int confidence_idx;
            int class_idx;
            int box_idx;
            int num_boxes_idx;
        }outmap_;
        

        cv::Mat preprocess( cv::Mat& image ) {
            
            // cv::Mat chw_image = hwc_to_chw(image);
            cv::Mat resizedImage;
            cv::resize(image, resizedImage, cv::Size(model_input_height, model_input_width), cv::INTER_LINEAR);            

            // Convert image to float32 and normalize
            cv::Mat floatImage;
            //resizedImage.convertTo(floatImage, CV_32F, 1.0 / 127.5, -1.0);
            resizedImage.convertTo(floatImage, CV_32F);

            return floatImage;
        }

        void draw_bounding_box(cv::Mat& image, std::vector<detectedObj>& detections_vector ){
            for(int i=0;i<num_boxes;++i ) {
                detectedObj detected_object = detections_vector[i];
                if(detected_object.accuracy>conf_thresh && detected_object.obj_id<81) { // Threshold, can be made function parameter
                    cv::rectangle(image, cv::Point(detected_object.x1, detected_object.y1), cv::Point(detected_object.x2, detected_object.y2), cv::Scalar(0, 255, 0), 2);

                    cv::putText(image, class_names.at( detected_object.obj_id ),
                                cv::Point(detected_object.x1, detected_object.y1 - 3), cv::FONT_ITALIC,
                                0.8, cv::Scalar(255, 255, 255), 2);

                    cv::putText(image, std::to_string(detected_object.accuracy),
                                cv::Point(detected_object.x1, detected_object.y1+30), cv::FONT_ITALIC,
                                0.8, cv::Scalar(255, 255, 0), 2);
                }
            }
        }

        std::vector<detectedObj> get_detections(std::vector<float*> output){
            std::vector<detectedObj> detections;
            detections.reserve(num_boxes);
            for (int i = 0; i < num_boxes; i++) {

                float confidence        = output[outmap_.confidence_idx][i];
                float x1                = output[outmap_.box_idx][i * 4+1];
                float y1                = output[outmap_.box_idx][i * 4];
                float x2                = output[outmap_.box_idx][i * 4 + 3];
                float y2                = output[outmap_.box_idx][i * 4 + 2];
                int classPrediction     = output[outmap_.class_idx][i];
                //printf("classPrediction: %d, confidence: %f, x1: %f, y1: %f, x2: %f, y2: %f\n", classPrediction, confidence, x1, y1, x2, y2);

                // Coords should be scaled to the original image. The coords from the model are relative to the model's input height and width.
                x1 = x1 * input_image_width ;
                x2 = x2 * input_image_width ;
                y1 = y1 * input_image_height ;
                y2 = y2 * input_image_height ;

                detectedObj obj( x1, x2, y1, y2, classPrediction, confidence);

                detections.push_back( obj );
            }
            return detections;
        }

        bool incallback_getframe(vector<const MX::Types::FeatureMap*> dst, int streamLabel){

            if(runflag.load()){
                cv::Mat inframe;
                cv::Mat rgbImage;
                bool got_frame = vcap.read(inframe);

                if (!got_frame) {
                    std::cout << "No frame \n\n\n";
                    return false;  // return false if frame retrieval fails
                }
                {
                    std::lock_guard<std::mutex> ilock(frame_queue_mutex);
                    cv::cvtColor(inframe, rgbImage, cv::COLOR_BGR2RGB);
                    frames_queue.push_back(rgbImage);
                }
                // Preprocess frame
                cv::Mat preProcframe = preprocess(rgbImage);

                if(type_ == App_Onnx){
                    // For ONNX models, we need to convert the image to CHW format
                    cv::Mat chwImage;
                    cv::dnn::blobFromImage(preProcframe, chwImage, 1.0, cv::Size(model_input_width, model_input_height), cv::Scalar(0, 0, 0), true, false);
                    preProcframe = chwImage;
                }

                dst[0]->set_data((float*)preProcframe.data);

                return true;
            }           
            else{
                vcap.release();
                return false;
            }    
        }

        bool outcallback_getmxaoutput(vector<const MX::Types::FeatureMap*> src, int streamLabel){

            for(int i =0; i<src.size();++i){
                src[i]->get_data(output[i]);
            }
            {
                std::lock_guard<std::mutex> ilock(frame_queue_mutex);
                // pop from frame queue
                displayImage = frames_queue.front();
                frames_queue.pop_front();
            }// releases in frame queue lock

            //Get the detections from model output
            num_boxes = output[outmap_.num_boxes_idx][0];
            //printf("num_boxes: %d\n", num_boxes);
            std::vector<detectedObj> detected_objectVector = get_detections(output);
            
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
        CenterNet(MX::Runtime::MxAccl* accl, std::string video_src, MxQt* gui, AppType type): type_{type}{
            //Assigning gui variable to class specifc variable
            gui_ = gui;
            //The output order is different for each type of pre-post processing. This is CenterNet specific.
            //To verify the output order we can view the centernet_post model's outputs.
            if(type_==App_Onnx){
                outmap_ = {.confidence_idx=0,.class_idx=1,.box_idx=3,.num_boxes_idx=2};
            }
            else if(type_==App_Tf){
                outmap_ = {.confidence_idx=3,.class_idx=4,.box_idx=5,.num_boxes_idx=2};
            }
            else{ // tflite
                outmap_ = {.confidence_idx=0,.class_idx=2,.box_idx=3,.num_boxes_idx=1};
            }
            // If the input is a camera, try to use optimal settings
            if(video_src.substr(0,3) == "cam"){
                src_is_cam = true;
                #ifdef __linux__
                    if (!openCamera(vcap, video_src[4]-'0', cv::CAP_V4L2)) {
                        throw(std::runtime_error("Failed to open: "+video_src));
                    }

                #elif defined(_WIN32)
                    if (!openCamera(vcap, video_src[4]-'0', cv::CAP_ANY)) {
                        throw(std::runtime_error("Failed to open: "+video_src));
                    }
                #endif
            }
            else if(video_src.substr(0,3) == "vid"){
                vcap.open(video_src.substr(4),cv::CAP_ANY);
                src_is_cam = false;
            }
            else{
                throw(std::runtime_error("Given video src: "+video_src+" is invalid"+
                "\n\n\tUse ./Centernet [cam:<camera index>|vid:<path to video file>] \n\n"));
            }
            if(!vcap.isOpened()){
                throw std::runtime_error("unknown input source passed");
            }

            // Getting input image dimensions
            input_image_width = static_cast<int>(vcap.get(cv::CAP_PROP_FRAME_WIDTH));
            input_image_height = static_cast<int>(vcap.get(cv::CAP_PROP_FRAME_HEIGHT));

            post_model_info = accl->get_post_model_info(0);
            for(int i=0;i<post_model_info.num_out_featuremaps;++i){
                output.push_back(new float[post_model_info.out_featuremap_sizes[i]]);
            } 

            model_info = accl->get_model_info(0);
            //Getting model input shapes and display size
            model_input_width = model_info.in_featuremap_shapes[0][0];
            model_input_height = model_info.in_featuremap_shapes[0][1];

            auto in_cb = std::bind(&CenterNet::incallback_getframe, this, std::placeholders::_1, std::placeholders::_2);
            auto out_cb = std::bind(&CenterNet::outcallback_getmxaoutput, this, std::placeholders::_1, std::placeholders::_2);
            accl->connect_stream(in_cb, out_cb, 0, 0);
            runflag.store(true);
        }
        ~CenterNet(){
            for(size_t i = 0; i< post_model_info.num_out_featuremaps;++i)
                delete[] output[i];
        }
};

int main(int argc, char *argv[]){
    signal(SIGINT, signal_handler);
    std::string video_src;

    MX::Runtime::MxAccl* accl;
    std::string plugin_name = std::string("onnx");
    if(argc==3){
        plugin_name = std::string(argv[1]);
        //Decoding the plugin passed by user
        if(plugin_name=="onnx"){
            //Create the Accl object and load the DFP
            accl = new MX::Runtime::MxAccl(onnx_model_path.c_str(), {0}, {true,true}, false, {20, 0, false, 12, 12}, {false, 0});
            //Connecting the pre-processing and post-processing models
            accl->connect_pre_model(onnx_preprocessing_model_path,0);
            accl->connect_post_model(onnx_postprocessing_model_path,0);
        }
        else if(plugin_name=="tf"){
            //Create the Accl object and load the DFP
            accl = new MX::Runtime::MxAccl(tf_model_path.c_str(), {0}, {true,true}, false, {20, 0, false, 12, 12}, {false, 0});
            //Connecting the pre-processing and post-processing models
            accl->connect_pre_model(tf_preprocessing_model_path,0);
            accl->connect_post_model(tf_postprocessing_model_path,0);
        }
        else if(plugin_name=="tflite"){
            //Create the Accl object and load the DFP
            accl = new MX::Runtime::MxAccl(tflite_model_path.c_str(), {0}, {true,true}, false, {20, 0, false, 12, 12}, {false, 0});
            //Connecting the pre-processing and post-processing models
            accl->connect_pre_model(tflite_preprocessing_model_path,0);
            accl->connect_post_model(tflite_postprocessing_model_path,0);
        }
        else{
            throw(std::runtime_error("Invalid pre-post plugin, "+plugin_name+" passed. Valid options are onnx,tf,tflite"
                "\n\n\tExample Use ./Centernet onnx [cam:<camera index>|vid:<path to video file>]\n\n"));            
        }
        //Decoding the custom input passed by user
        std::string video_str(argv[2]);
        video_src = video_str;
    }
    else{
        throw(std::runtime_error("Need to pass exactly two input arguments. Valid options are onnx,tf,tflite"
            "\n\n\tExample Use ./Centernet onnx [cam:<camera index>|vid:<path to video file>]\n\n"));            
    }
    // Creating GuiView which is a memryx qt util for easy display
    MxQt gui(argc,argv);
    // Setting the layout of the display based on number of input streams
    gui.screens[0]->SetSquareLayout(1,false);


    //Creating a CenterNet object for each stream which also connects the corresponding stream to accl.
    CenterNet* obj;
    if(plugin_name=="onnx"){
        obj = new CenterNet(accl,video_src,&gui,App_Onnx);
    }
    else if (plugin_name=="tf"){
        obj = new CenterNet(accl,video_src,&gui,App_Tf);
    }
    else{
        obj = new CenterNet(accl,video_src,&gui,App_Tflite);
    }
    //Run the accelerator and wait
    accl->start();
    gui.Run();  //This command waits for exit to be pressed in Qt window
    accl->stop();

    delete accl;
    delete obj;
}
