#pragma once

#include <vector>

typedef void *mxutil_cam_t;

typedef enum
{
    mxutil_IMG_FMT_MJPG = 1,
    mxutil_IMG_FMT_RGB24, // RGB888
    mxutil_IMG_FMT_GREY,
    mxutil_IMG_FMT_YUYV, // 16 bit per pixel
    mxutil_IMG_FMT_OTHERS,
} mxutil_cam_pixel_format_e;

typedef struct
{
    int width;
    int height;
    mxutil_cam_pixel_format_e pixfmt;
} mxutil_cam_setting_t;

// return a vector reporting which /dev/video%d is supported
// FIXME: for now only allow MJPG and YUYV cameras
std::vector<int> mxutil_cam_filter_supported();

// open a camera, for Linux, it use V4L2, it reads /dev/video# for cam_id
// not support Windows yet
mxutil_cam_t mxutil_cam_open(int cam_id);

// get camera settings
int mxutil_cam_get_setting(mxutil_cam_t cc, mxutil_cam_setting_t *cc_setting);

// read frame buffer pointer
void *mxutil_cam_get_frame(mxutil_cam_t cc);

// return frame buffer pointer
int mxutil_cam_put_frame(mxutil_cam_t cc, void *frame_buf);

// close the camera
int mxutil_cam_close(mxutil_cam_t cc);
