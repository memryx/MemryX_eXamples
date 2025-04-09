#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <thread>
#include <chrono>
#include <cstdint>
#include <vector>
#include <unistd.h>

#include <opencv2/opencv.hpp>

#include "fifo_queue.h"
#include "vdo_predec.h"

class _mxutil_vdo_player_h
{
public:
    std::vector<void *> frames;
    std::chrono::milliseconds last_frame_ms;
    int next_frame_idx;
    int frame_intv;
    int org_frame_width = 0;
    int org_frame_height = 0;
};

class _mxutil_vdo_player_xxx_h
{
public:
    int disp_width = 0;
    int disp_height = 0;
    void *frame_buf;
    mxutil_fifo_queue<void *> frame_bufs;
    cv::VideoCapture cap;
};

mxutil_vdo_player_real_h mxutil_vdo_player_real(const char *vdo_file_path, int disp_width, int disp_height)
{
    _mxutil_vdo_player_xxx_h *_vpctx = new _mxutil_vdo_player_xxx_h;
    cv::VideoCapture &cap = _vpctx->cap;

    if (!cap.open(vdo_file_path))
    {
        std::cerr << "Error: Could not open the video file." << std::endl;
        exit(-1);
    }

    // Get video properties
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    std::cout << "Video resolution: " << frameWidth << "x" << frameHeight << " @ " << fps << " FPS" << std::endl;
    // printf("%p %p %p\n", _vpctx->cap, _vpctx->disp_width, _vpctx->disp_height);
    _vpctx->disp_width = disp_width;
    _vpctx->disp_height = disp_height;
    
    return (mxutil_vdo_player_real_h) _vpctx;
}

void *mxutil_vdo_player_get_frame_real(mxutil_vdo_player_real_h vh)
{
    _mxutil_vdo_player_xxx_h *_vpctx = (_mxutil_vdo_player_xxx_h *)vh;
    int disp_width = _vpctx->disp_width;
    int disp_height = _vpctx->disp_height;
    int disp_frame_size = disp_width * disp_height * 3;

    cv::VideoCapture &cap = _vpctx->cap;
    cv::Mat frame;

    _vpctx->cap.read(frame);
    // Read a new frame from the video
    while (!cap.read(frame))
    {
        // If reading fails, restart the video
        // std::cout << "Restarting video..." << std::endl;
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        continue;
    }
    _vpctx->frame_buf = malloc(disp_frame_size);

    if (_vpctx->frame_buf == NULL)
    {
        printf("warning !! memory not enough for video frame decoding !!\n");
        exit(0);
    }

    _vpctx->frame_bufs.push(_vpctx->frame_buf);
    cv::Mat resized_frame(disp_height, disp_width, CV_8UC3, _vpctx->frame_buf);
    cv::resize(frame, resized_frame, cv::Size(disp_width, disp_height), cv::INTER_LINEAR);
    cv::cvtColor(resized_frame, resized_frame, cv::COLOR_BGR2RGB);

    return _vpctx->frame_buf;
}

void mxutil_vdo_player_return_frame_real(mxutil_vdo_player_real_h vh)
{
    _mxutil_vdo_player_xxx_h *_vpctx = (_mxutil_vdo_player_xxx_h *)vh;
    if (_vpctx->frame_bufs.empty())
        return;

    void *frame_buf = _vpctx->frame_bufs.pop();
    if (frame_buf != NULL)
    {
        free(frame_buf);
        frame_buf = NULL;
    }
}

void mxutil_vdo_player_close_real(mxutil_vdo_player_real_h vh)
{
    _mxutil_vdo_player_xxx_h *_vpctx = (_mxutil_vdo_player_xxx_h *)vh;

    while (!_vpctx->frame_bufs.empty())
    {
        void *frame_buf = _vpctx->frame_bufs.pop();
        _vpctx->frame_bufs.pop();
        if (frame_buf != NULL)
        {
            free(frame_buf);
            frame_buf = NULL;
        }
    }

    _vpctx->cap.release();
    delete _vpctx;
}

void mxutil_vdo_player_get_frame_resolution(mxutil_vdo_player_h vh, int &width, int &height)
{
    _mxutil_vdo_player_h *_vpctx = (_mxutil_vdo_player_h *)vh;
    width = _vpctx->org_frame_width;
    height = _vpctx->org_frame_height;
}

mxutil_vdo_player_h mxutil_vdo_player_decode(const char *vdo_file_path, int num_frames, int resized_width, int resized_height, int frame_fmt, int fps)
{
    _mxutil_vdo_player_h *_vpctx = new _mxutil_vdo_player_h;
    _vpctx->frames.clear();
    _vpctx->last_frame_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    _vpctx->next_frame_idx = 0;

    cv::VideoCapture vcap;
    vcap.open(vdo_file_path);
    if (!vcap.isOpened())
    {
        printf("open %s failed\n", vdo_file_path);
        return NULL;
    }

    // int fps = vcap.get(cv::CAP_PROP_FPS);
    // fps = 30;
    int org_frame_width = vcap.get(cv::CAP_PROP_FRAME_WIDTH);
    int org_frame_height = vcap.get(cv::CAP_PROP_FRAME_HEIGHT);
    _vpctx->frame_intv = (int)(1000 / fps);
    _vpctx->org_frame_width = org_frame_width;
    _vpctx->org_frame_height = org_frame_height;

    int resized_frame_size = resized_width * resized_height * 3;

    printf("decoding %s, resolution = %dx%d, FPS = %d\n", vdo_file_path, org_frame_width, org_frame_height, fps);

    int i;
    for (i = 0; i < num_frames; i++)
    {
        cv::Mat decoded_frame;
        if (!vcap.read(decoded_frame))
        {
            break;
        }

        void *frame_buf = malloc(resized_frame_size);
        if (frame_buf == NULL)
        {
            printf("warning !! memory not enough for video frame decoding !!\n");
            exit(0);
        }

        cv::Mat resized_frame(resized_height, resized_width, CV_8UC3, frame_buf);
        cv::resize(decoded_frame, resized_frame, cv::Size(resized_width, resized_height), cv::INTER_LINEAR);
        if (frame_fmt == FRAME_FMT_RGB)
            cv::cvtColor(resized_frame, resized_frame, cv::COLOR_BGR2RGB);

        _vpctx->frames.push_back(frame_buf);
    }

    // printf("decoded %d frames from %s\n", i, vdo_file_path);
    vcap.release();
    return (mxutil_vdo_player_h)_vpctx;
}

void *mxutil_vdo_player_get_frame(mxutil_vdo_player_h vh)
{
    _mxutil_vdo_player_h *_vpctx = (_mxutil_vdo_player_h *)vh;

    std::chrono::milliseconds now_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

    if (_vpctx->next_frame_idx >= (int)_vpctx->frames.size())
        _vpctx->next_frame_idx = 0;

    void *frame = _vpctx->frames.at(_vpctx->next_frame_idx);

    // speed control
    int diff_ms = (int)now_ms.count() - (int)_vpctx->last_frame_ms.count();
    int sleep_ms = _vpctx->frame_intv - diff_ms;
    if (sleep_ms > 0)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
    }

    _vpctx->next_frame_idx++;
    _vpctx->last_frame_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

    return frame;
}

void mxutil_vdo_player_close(mxutil_vdo_player_h vh)
{
    _mxutil_vdo_player_h *_vpctx = (_mxutil_vdo_player_h *)vh;

    for (int i = 0; i < (int)_vpctx->frames.size(); i++)
    {
        free(_vpctx->frames.at(i));
    }

    _vpctx->frames.clear();
    delete _vpctx;
}
