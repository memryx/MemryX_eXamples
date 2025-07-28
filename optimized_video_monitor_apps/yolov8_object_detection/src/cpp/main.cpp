/*
╔═════════════════════ Main Execution Loop ═══════════════════════════════╗

   ┌─────────────────────── Input Callback ────────────────────────────┐
   │                                                                   │
   │   Retrieves frames                                                │
   │   Pre-processes frames                                            │
   │   Sends data to accelerator                                       │
   │                                                                   │
   └───────────────────────────────────────────────────────────────────┘
                                🔽
                                🔽
                                🔽
   ┌─────────────────────── Output Callback ───────────────────────────┐
   │                                                                   │
   │   Receives data from accelerator                                  │
   │   Applies post-processing                                         │
   │   Draws detection and renders output                              │
   │                                                                   │
   └───────────────────────────────────────────────────────────────────┘

╚═════════════════════════════════════════════════════════════════════════╝
*/
#include <stdio.h>
#include <signal.h>
#include <thread>
#include <chrono>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include <memx/accl/MxAccl.h>

#include "utils/gui_view.h"
#include "utils/input_source.h"
#include "utils/vms.h"
#include "utils/yolov8.h"

constexpr int kMaxNumChannels = 100;
constexpr int kFpsCountMax = 120;
constexpr char kDefaultConfigPath[] = "assets/config.txt";

struct ChannelObject
{
    InputSource *input_source; // input capture device, could be usb-cam, ip-cam, or video
    DisplayScreen *screen;     // associated with a specific screen
    uint32_t disp_width;
    uint32_t disp_height;
    int frame_count;
    float fps_number;
    std::chrono::milliseconds start_ms;
    std::unique_ptr<YOLOv8> yolov8_handle;
    MX::Utils::fifo_queue<cv::Mat *> disp_frames; // sync queue between in/out callback
};

// Global variables
ChannelObject g_chan_objs[kMaxNumChannels];
VmsCfg g_config;
vector<InputSource *> g_input_sources;
MX::Types::MxModelInfo g_model_info;
MX::Utils::fifo_queue<std::vector<float *>> g_input_data_buf;
MX::Utils::fifo_queue<std::vector<float *>> g_output_data_buf;
std::atomic<uint64_t> g_frame_count(0);
int g_duration_in_secs = 5;
bool g_is_running = true;
YOLOGuiView *g_gui = NULL;

void signalHandler(int /*pSignal*/)
{
    g_is_running = false;
    if (g_gui)
        g_gui->Exit();
}

void AllocateMemory(MX::Types::MxModelInfo &model_info, std::vector<float *> *pInput_data, std::vector<float *> *pOutput_data)
{
    for (int i = 0; i < model_info.num_in_featuremaps; ++i)
    {
        float *pData = new float[model_info.in_featuremap_sizes[i]];
        memset(pData, 0, model_info.in_featuremap_sizes[i] * sizeof(float));
        pInput_data->push_back(pData);
    }
    for (int i = 0; i < model_info.num_out_featuremaps; ++i)
    {
        float *pData = new float[model_info.out_featuremap_sizes[i]];
        memset(pData, 0, model_info.out_featuremap_sizes[i] * sizeof(float));
        pOutput_data->push_back(pData);
    }
}

float CalculateFPS(ChannelObject &channel)
{
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()) -
                    channel.start_ms;
    float frames_per_second = static_cast<float>(kFpsCountMax) * 1000.0f / duration.count();
    channel.fps_number = frames_per_second;
    channel.frame_count = 0;
    return frames_per_second;
}

float UpdatedFPS(int idx)
{
    auto &chan_obj = g_chan_objs[idx];
    ++chan_obj.frame_count;
    ++g_frame_count;

    if (chan_obj.frame_count == 1)
    {
        chan_obj.start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch());
    }
    else if (chan_obj.frame_count == (kFpsCountMax + 1))
    {
        return CalculateFPS(chan_obj);
    }

    return 0.0f;
}

bool incallback_func(vector<const MX::Types::FeatureMap *> dst, int channel_idx)
{
    if (!g_is_running)
        return false;

    int idx = channel_idx;
    auto &chan_obj = g_chan_objs[idx];
    auto &input_source = chan_obj.input_source;
    auto &screen = chan_obj.screen;
    uint32_t disp_width = chan_obj.disp_width;
    uint32_t disp_height = chan_obj.disp_height;
    auto yolov8_handle = chan_obj.yolov8_handle.get();

    // Compute letterbox padding for YOLO model
    yolov8_handle->ComputePadding(disp_width, disp_height);

    // Retrieve disp_frame buffer of size (disp_width, disp_height)
    cv::Mat *disp_frame = screen->GetDisplayFrameBuf(idx);

    // Fill frame from input source
    // TODO: decouple the input and display resolution
    input_source->GetFrame(*disp_frame);

    // Retrieve accl input buffer and perform preprocessing
    std::vector<float *> accl_input_data = g_input_data_buf.pop();
    yolov8_handle->PreProcess(disp_frame->data, disp_width, disp_height, accl_input_data);

    // Set preprocessed input data for accelerator
    for (int in_idx = 0; in_idx < g_model_info.num_in_featuremaps; ++in_idx)
    {
        dst[in_idx]->set_data(accl_input_data[in_idx]);
    }

    // Push the frame into queue for rendering in output callback
    chan_obj.disp_frames.push(disp_frame);

    // Push the input data buffer back for reuse
    g_input_data_buf.push(accl_input_data);

    return true;
}

bool outcallback_func(vector<const MX::Types::FeatureMap *> src, int channel_idx)
{
    if (!g_is_running)
        return false;

    int idx = channel_idx;
    auto &chan_obj = g_chan_objs[idx];
    auto &screen = chan_obj.screen;
    auto yolov8_handle = chan_obj.yolov8_handle.get();
    float confidence = (screen->GetConfidenceValue() == -1.0) ? g_config.inf_confidence : screen->GetConfidenceValue();
    YOLOv8Result result;

    // Retrieve the frame from display queue
    cv::Mat *disp_frame = chan_obj.disp_frames.pop();

    // Retrieve output buffer and process the inference results
    std::vector<float *> accl_output_data = g_output_data_buf.pop();

    // Retrieve output data from accelerator
    for (int out_idx = 0; out_idx < g_model_info.num_out_featuremaps; ++out_idx)
    {
        src[out_idx]->get_data(accl_output_data[out_idx]);
    }

    // Set confidence threshold and process detection results
    yolov8_handle->SetConfidenceThreshold(confidence);
    yolov8_handle->PostProcess(accl_output_data, result);
    yolov8_handle->DrawResult(result, *disp_frame);

    // FPS Calculation
    float fps_number = UpdatedFPS(idx);

    // Push output buffer back for reuse
    g_output_data_buf.push(accl_output_data);

    // Set frame to display with FPS overlay
    screen->SetDisplayFrame(idx, disp_frame, fps_number);

    return true;
}

void Clean()
{
    while (g_input_data_buf.size() != 0)
    {
        std::vector<float *> input_data = g_input_data_buf.pop();
        for (auto ptr : input_data)
        {
            if (ptr)
                delete ptr;
            ptr = nullptr;
        }
    }

    while (g_output_data_buf.size() != 0)
    {
        std::vector<float *> output_data = g_output_data_buf.pop();
        for (auto ptr : output_data)
        {
            if (ptr)
                delete ptr;
            ptr = nullptr;
        }
    }

    for (long unsigned int i = 0; i < g_input_sources.size(); i++)
    {
        if (g_input_sources.at(i))
            delete g_input_sources.at(i);
    }
}

pair<long, long> GetCPUTimes()
{
    ifstream stat_file("/proc/stat");
    string line;
    getline(stat_file, line);
    stat_file.close();

    vector<long> times;
    string cpu;
    long value;
    istringstream iss(line);
    iss >> cpu; // Skip "cpu" field
    while (iss >> value)
    {
        times.push_back(value);
    }

    long idle_time = times[3]; // idle time (4th field)
    long total_time = accumulate(times.begin(), times.end(), 0L);

    return {idle_time, total_time};
}

double CalculateCPULoad(pair<long, long> prev, pair<long, long> current)
{
    long idle_diff = current.first - prev.first;
    long total_diff = current.second - prev.second;
    return 100.0 * (1.0 - static_cast<double>(idle_diff) / total_diff);
}

void InfoWatcher(int monitoring_duration_seconds)
{
    if (monitoring_duration_seconds <= 0)
    {
        std::cerr << "Error: monitoring_duration_seconds must be greater than 0" << std::endl;
        return;
    }

    uint64_t prev_frame_count = g_frame_count;
    pair<long, long> prev_times = GetCPUTimes();
    int idx_print = 0;
    int run_count = 0;
    unsigned int sleep_duration_ms = 100;
    int target_count = monitoring_duration_seconds * 1000 / sleep_duration_ms;
    while (g_is_running)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_duration_ms));
        run_count++;
        if (run_count == target_count)
        {
            {
                pair<long, long> current_times = GetCPUTimes();
                double cpu_load = CalculateCPULoad(prev_times, current_times);
                prev_times = current_times;

                // FPS
                uint64_t diff_count = g_frame_count - prev_frame_count;
                prev_frame_count = g_frame_count;
                printf("%d: FPS %.1f | CPU_load %.1f %%\n", idx_print++, (float)diff_count / monitoring_duration_seconds, cpu_load);
            }
            run_count = 0;
        }
    }
    return;
}

void InitScreen(YOLOGuiView &gui, int screen_idx)
{
    auto screen = gui.screens.at(screen_idx).get();
    screen->SetSquareLayout(g_config.num_chs);           // Specify num_chs as 4 for a 2x2 square layout.
    screen->SetConfidenceValue(g_config.inf_confidence); // Add this if the demo needs to modify confidence at runtime
    if (!g_config.model_name.empty())
        screen->SetModelName(g_config.model_name);
    g_gui = &gui;
}

void InitChannelObjects(YOLOGuiView &gui, int idx)
{
    // Using a global object because MX_API does not support passing custom arguments into callback functions yet.
    int model_input_height = g_model_info.in_featuremap_shapes[0][0];
    int model_input_width = g_model_info.in_featuremap_shapes[0][1];
    int model_input_channel = g_model_info.in_featuremap_shapes[0][3];

    auto screen = gui.screens.at(g_config.screen_idx).get();
    g_chan_objs[idx].screen = screen;
    g_chan_objs[idx].disp_width = screen->GetViewerWidth(idx);
    g_chan_objs[idx].disp_height = screen->GetViewerHeight(idx);
    g_chan_objs[idx].frame_count = 0;
    g_chan_objs[idx].input_source = g_input_sources.at(idx);
    g_chan_objs[idx].yolov8_handle = std::make_unique<YOLOv8>(
        model_input_width, model_input_height, model_input_channel,
        g_config.inf_confidence, g_config.inf_iou);
}

void AllocateBuffers()
{
    for (int i = 0; i < 3 * g_config.num_chs; i++)
    {
        std::vector<float *> input_data;
        std::vector<float *> output_data;
        AllocateMemory(g_model_info, &input_data, &output_data);
        g_input_data_buf.push(input_data);
        g_output_data_buf.push(output_data);
    }
}

void ParseArgs(int argc, char *argv[])
{
    int opt;
    string config_path;
    while ((opt = getopt(argc, argv, "c:d:h")) != -1)
    {
        switch (opt)
        {
        case 'c':
            config_path = string(optarg);
            ReadVmsConfigFromFile(config_path.c_str(), g_config);
            break;
        case 'd':
            g_duration_in_secs = std::stoi(optarg);
            break;
        case 'h':
        default:
            printf("-c: config file for the demo,\t\t\tdefault: %s\n", kDefaultConfigPath);
            printf("-d: duration to measure FPS and CPU loading,\tdefault: 5 seconds\n");
            exit(1);
            break;
        }
    }
    if (config_path.empty())
    {
        ReadVmsConfigFromFile(kDefaultConfigPath, g_config);
    }
    return;
}

int main(int argc, char *argv[])
{
    // Handle SIGINT (Ctrl+C) signal
    signal(SIGINT, signalHandler);

    // Parse command-line arguments
    ParseArgs(argc, argv);

    std::vector<int> device_ids = {0};
    std::array<bool, 2> use_model_shape = {false, false};

    // Create the accelerator object and load the DFP model
    std::unique_ptr<MX::Runtime::MxAccl> accl = std::make_unique<MX::Runtime::MxAccl>(filesystem::path(g_config.dfp_file),device_ids, use_model_shape);

    int model_id = 0; // The DFP is compiled with a single model
    g_model_info = accl->get_model_info(model_id);

    // Allocate input and output buffer pools used in callback functions
    AllocateBuffers();

    // Initialize GUI and specify the screen for display
    YOLOGuiView gui(argc, argv);
    InitScreen(gui, g_config.screen_idx);
    auto screen = gui.screens.at(g_config.screen_idx).get();

    // Initialize input capture sources
    InitCaps(screen, g_config, g_input_sources);

    for (uint32_t channel_idx = 0; channel_idx < screen->NumViewers(); channel_idx++)
    {
        InitChannelObjects(gui, channel_idx);

        // Connect streams to the accelerator
        accl->connect_stream(
            &incallback_func,  // Input callback function
            &outcallback_func, // Output callback function
            channel_idx,       // Unique stream ID
            model_id           // Model ID
        );
    }

    // Limit the number of threads to prevent huge CPU usage
    int num_cpu = std::thread::hardware_concurrency();
    int num_streams = screen->NumViewers();
    int oworkers = std::max(1, std::min(num_streams, (num_cpu / 4)+1));
    int iworkers = std::max(1, std::min(num_streams, num_cpu / 6));
    printf("num_cpu %d, num_streams %d, oworkers %d, iworkers %d\n",
            num_cpu, num_streams, oworkers, iworkers);
    accl->set_num_workers(iworkers,oworkers);

    // Start the accelerator after connecting streams
    accl->start();

    // Start a separate thread to watch for runtime info
    std::thread info_watcher = std::thread(InfoWatcher, g_duration_in_secs);

    // Run GUI (blocks main thread until exit button is pressed)
    printf("press exit button (at top right) or ctrl-c to exit\n");

    screen->Show();
    gui.Run();

    // Mark the application as not running
    g_is_running = false;

    // Stop the accelerator and wait for completion
    accl->stop();
    info_watcher.join();

    // Cleanup allocated resources
    Clean();

    printf("Exit successfully.\n");

    return 0;
}
