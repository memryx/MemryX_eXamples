#pragma once

#include <string>
#include <vector>
#include <fstream>

#include "gui_view.h"
#include "input_source.h"

using namespace std;

typedef struct
{
    int num_chs;
    int video_predecoded_frames;
    int screen_idx;
    std::vector<int> group_id;
    float inf_confidence;
    float inf_iou;
    std::string dfp_file;
    std::string logo_file;
    std::string model_name;
    std::vector<VideoInputSource_s> video_inputs;
} VmsCfg;

void ReadVmsConfigFromFile(const char *cfg_path, VmsCfg &config);
void InitCapFunc(VmsCfg config, int idx, InputSource **stream_cap, int disp_width, int disp_height);
void InitCaps(DisplayScreen *screen, VmsCfg &config, vector<InputSource *> &caps);
