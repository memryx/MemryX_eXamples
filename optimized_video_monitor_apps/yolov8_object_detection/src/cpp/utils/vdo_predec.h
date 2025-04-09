#pragma once

typedef void *mxutil_vdo_player_h;
typedef void *mxutil_vdo_player_real_h;

#define FRAME_FMT_BGR (1)
#define FRAME_FMT_RGB (2)

mxutil_vdo_player_h mxutil_vdo_player_decode(const char *vdo_file_path, int num_frames, int resized_width, int resized_height, int frame_fmt = FRAME_FMT_BGR, int target_fps = 30);
void *mxutil_vdo_player_get_frame(mxutil_vdo_player_h vfctx);
void mxutil_vdo_player_close(mxutil_vdo_player_h vh);
void mxutil_vdo_player_get_frame_resolution(mxutil_vdo_player_h video_ctx_, int &width, int &height);


mxutil_vdo_player_real_h mxutil_vdo_player_real(const char *vdo_file_path, int disp_width, int disp_height);
void *mxutil_vdo_player_get_frame_real(mxutil_vdo_player_real_h vh);
void mxutil_vdo_player_return_frame_real(mxutil_vdo_player_real_h vh);
void mxutil_vdo_player_close_real(mxutil_vdo_player_real_h vh);