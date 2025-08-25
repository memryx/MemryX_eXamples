#include "vms.h"

void ReadVmsConfigFromFile(const char *cfg_path, VmsCfg &config)
{
    std::ifstream infile;

    // some defaults
    config.num_chs = 16;
    config.inf_confidence = 0.3;
    config.inf_iou = 0.45;
    config.screen_idx = 0;
    printf("reading config = %s\n", cfg_path);
    infile.open(cfg_path);
    if (infile.is_open())
    {
        std::string param;
        std::string value;
        while (getline(infile, param, '='))
        {
            value.clear();

            if (std::getline(infile, value, '\r') || std::getline(infile, value, '\n'))
            {
                if (!infile.eof() && infile.peek() == '\n')
                    infile.ignore();
            }

            if (value.empty())
                continue;

            if (param == string("num_chs"))
            {
                config.num_chs = stoi(value);
                printf("(VMS config) num of CHs = %d\n", config.num_chs);
                if (config.num_chs <= 0)
                {
                    printf("invalid num of CHs in vms config file, please check.\n");
                    exit(0);
                }
            }
            else if (param == string("video_predecoded_frames"))
            {
                config.video_predecoded_frames = stoi(value);
                // printf("(VMS config) video_predecoded_frames = %d\n", config.video_predecoded_frames);
            }
            else if (param == string("ip_cam"))
            {
                VideoInputSource_s vis = {VIDEO_FROM_IPCAM, value};
                config.video_inputs.push_back(vis);
                // printf("(VMS config) ip cam = %s\n", value.c_str());
            }
            else if (param == string("video"))
            {
                VideoInputSource_s vis = {VIDEO_FROM_FILE, value};
                config.video_inputs.push_back(vis);
                // printf("(VMS config) video = %s\n", value.c_str());
            }
            else if (param == string("usb_cam"))
            {
                VideoInputSource_s vis = {VIDEO_FROM_USBCAM, value};
                config.video_inputs.push_back(vis);
                // printf("(VMS config) USB cam = %s\n", value.c_str());
            }
            else if (param == string("dfp"))
            {
                VideoInputSource_s vis = {VIDEO_FROM_USBCAM, value};
                config.dfp_file = value;
                printf("(VMS config) dfp = %s\n", value.c_str());
            }
            else if (param == string("model_name"))
            {
                VideoInputSource_s vis = {VIDEO_FROM_USBCAM, value};
                config.model_name = value;
                printf("(VMS config) model name = %s\n", value.c_str());
            }
            else if (param == string("group"))
            {
                config.group_id.push_back(stoi(value));
            }
            else if (param == string("logo"))
            {
                config.logo_file = value;
                // printf("(VMS config) logo image = %s\n", value.c_str());
            }
            else if (param == string("inf_confidence"))
            {
                config.inf_confidence = stof(value);
                // printf("(VMS config) confidence = %.2f \n", config.inf_confidence);
            }
            else if (param == string("inf_iou"))
            {
                config.inf_iou = stof(value);
                // printf("(VMS config) IOU = %.2f \n", config.inf_iou);
            }
            if (param == string("screen_idx"))
            {
                config.screen_idx = stoi(value);
                printf("(VMS config) use screen idx = %d\n", config.screen_idx);
                if (config.screen_idx < 0)
                {
                    printf("invalid screen id.\n");
                    exit(0);
                }
            }
        }
    }
    else
    {
        printf("cannot find config file %s\n", cfg_path);
        exit(EXIT_FAILURE);
    }
}

void InitCapFunc(VmsCfg config, int idx, InputSource **stream_cap, int disp_width, int disp_height)
{
    VideoInputSource_s &vis = config.video_inputs.at(idx);

    if (vis.type == VIDEO_FROM_IPCAM)
    {
        *stream_cap = new IpCamStream(vis.access_value.c_str(), disp_width, disp_height);
    }
    else if (vis.type == VIDEO_FROM_FILE)
    {
        *stream_cap = new VideoFileStream(vis.access_value.c_str(), disp_width, disp_height, config.video_predecoded_frames, 60);
        // *stream_cap = new VideoFileStreamReal(vis.access_value.c_str(), disp_width, disp_height);
    }
    else if (vis.type == VIDEO_FROM_USBCAM)
    {
        int dev_fd = atoi(vis.access_value.c_str());
        *stream_cap = new UsbCamStream(dev_fd, disp_width, disp_height);
    }
}

void InitCaps(DisplayScreen *screen, VmsCfg &config, vector<InputSource *> &caps)
{
    // each screen contains numerous viewers, and each viewer should connect with a input source
    size_t num_viewers = screen->NumViewers();
    InputSource *stream_cap[num_viewers];
    std::thread threads[num_viewers];

    for (size_t idx = 0; idx < num_viewers; idx++)
    {
        if (idx >= config.video_inputs.size())
            break;

        int disp_width = screen->GetViewerWidth(idx);
        int disp_height = screen->GetViewerHeight(idx);

        threads[idx] = std::thread(InitCapFunc, config, idx, &(stream_cap[idx]), disp_width, disp_height);
    }

    for (size_t idx = 0; idx < num_viewers; idx++)
    {
        if (idx >= config.video_inputs.size())
            break;

        threads[idx].join();
        caps.push_back(stream_cap[idx]);
    }
}
