#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <thread>

#include "ipcam_stream.h"

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include "libavutil/avutil.h"
#include <libswscale/swscale.h>
}

#define FRAME_BUF_SIZE 3

using namespace std;

class _mxutil_stream_player_h
{
private:
    SwsContext *img_convert_ctx_ = NULL;
    AVDictionary *options_ = NULL;
    AVPacket packet_;
    const AVCodec *codec_;
    AVFrame *frame_yuv_;
    int video_stream_index_;
    int disp_width_, disp_height_;
    std::thread stream_thread_;

public:
    AVFormatContext *format_ctx_ = NULL;
    AVCodecContext *codec_ctx_ = NULL;
    string stream_source_name;
    int stream_frame_width_, stream_frame_height_;
    double frame_rate_;
    bool running_;

    AVFrame *buf = NULL;
    // bidirectional buffers used for queueing rgb frame
    fifo_queue<AVFrame *> available_frame_bufs_;
    fifo_queue<AVFrame *> frames_;

    _mxutil_stream_player_h(const char *stream_url, const int disp_width_, const int disp_height_);
    ~_mxutil_stream_player_h();
    void mxutil_stream_player_main_worker();
    void mxutil_stream_player_reconnect();
};

// Return the frame allocating by ffmpeg with specific frame info
// By default the ffmpeg will not allocate AVFrame and it's buffer at once.
AVFrame *av_frame_alloc_with_info(int format, int width, int height)
{
    AVFrame *frame = av_frame_alloc();
    frame->format = format;
    frame->width = width;
    frame->height = height;
    av_frame_get_buffer(frame, 0);

    return frame;
}

_mxutil_stream_player_h::~_mxutil_stream_player_h()
{
    // cleanup the worker thread
    stream_thread_.join();

    // release sources
    while (!available_frame_bufs_.empty())
        av_free(available_frame_bufs_.pop());
    while (!frames_.empty())
        av_free(frames_.pop());

    av_free(frame_yuv_);
    sws_freeContext(img_convert_ctx_);
    avcodec_close(codec_ctx_);
    avformat_close_input(&format_ctx_);
    std::cout << "Close " << stream_source_name << std::endl;
}

static int interrupt_callback(void *handle)
{
    _mxutil_stream_player_h *ctx = (_mxutil_stream_player_h *)handle;
    if (ctx->running_ == false)
    {
        // printf("%s interrupted to leave\n", ctx->stream_source_name.c_str());
        return 1;
    }
    return 0;
}

_mxutil_stream_player_h::_mxutil_stream_player_h(const char *stream_url, const int disp_width_, const int disp_height_)
{
    this->stream_source_name = string(stream_url);
    this->disp_width_ = disp_width_;
    this->disp_height_ = disp_height_;
    this->running_ = true;

    // Open the initial context variables that are needed
    format_ctx_ = avformat_alloc_context();

    format_ctx_->interrupt_callback.callback = interrupt_callback;
    format_ctx_->interrupt_callback.opaque = (this);

    // set tcp transport for rtsp
    av_dict_set(&options_, "stimeout", "7000000", 0);
    av_dict_set(&options_, "rtsp_transport", "tcp", 0);
    av_dict_set(&options_, "fflags", "nobuffer", 0);
    av_dict_set(&options_, "fflags", "flush_packets", 0);

    // av_register_all();

    // open RTSP
    if (avformat_open_input(&format_ctx_, stream_url, NULL, &options_) != 0)
    {
        throw std::runtime_error("Error: Failed to open RTSP input stream.");
    }
    std::cout << "Open RTSP stream " << stream_url << std::endl;

    video_stream_index_ = -1;

    if (avformat_find_stream_info(format_ctx_, NULL) < 0)
    {
        throw std::runtime_error("Error: Could not find stream information.");
    }

    // stream demuxer
    for (unsigned int i = 0; i < format_ctx_->nb_streams; i++)
    {
        if (format_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
        {
            video_stream_index_ = i;
            break;
        }
    }

    if (video_stream_index_ == -1)
    {
        throw std::runtime_error("Error: Could not find a video stream.");
    }

    // play RTSP
    av_read_play(format_ctx_);

    // Get the codec_
    codec_ = avcodec_find_decoder(format_ctx_->streams[video_stream_index_]->codecpar->codec_id);
    // codec_ = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (codec_ == NULL)
    {
        throw std::runtime_error("Error: Unsupported codec_.");
    }

    codec_ctx_ = avcodec_alloc_context3(codec_);
    if (avcodec_parameters_to_context(codec_ctx_, format_ctx_->streams[video_stream_index_]->codecpar) < 0)
    {
        throw std::runtime_error("Error: Could not copy codec_ parameters.");
    }

    if (avcodec_open2(codec_ctx_, codec_, NULL) < 0)
    {
        throw std::runtime_error("Error: Could not open codec_.");
    }

    stream_frame_width_ = format_ctx_->streams[video_stream_index_]->codecpar->width;
    stream_frame_height_ = format_ctx_->streams[video_stream_index_]->codecpar->height;
    frame_rate_ = av_q2d(format_ctx_->streams[video_stream_index_]->avg_frame_rate);

    printf("media info: resolution = %dx%d, FPS = %d\n", stream_frame_width_, stream_frame_height_, (int)frame_rate_);

    // get ffmpeg sws context to convert codec_ output to BGR
    // sws_scale needs width to be 32x on miniPC
    // padding is needed for memory alignment, and that the size of padding depends on CPU used
    img_convert_ctx_ = sws_getContext(codec_ctx_->width, codec_ctx_->height, codec_ctx_->pix_fmt,
                                      disp_width_, disp_height_, AV_PIX_FMT_RGB24,
                                      SWS_BICUBIC, NULL, NULL, NULL);

    // av_init_packet(&packet_);

    frame_yuv_ = av_frame_alloc_with_info(codec_ctx_->pix_fmt, stream_frame_width_, stream_frame_height_);

    for (int i = 0; i < FRAME_BUF_SIZE; i++)
    {
        AVFrame *buf = av_frame_alloc_with_info(AV_PIX_FMT_RGB24, disp_width_, disp_height_);
        if (buf == NULL)
        {
            throw std::runtime_error("Error: Failed to allocate memory.");
        }
        available_frame_bufs_.push(buf);
    }

    // creating worker thread
    stream_thread_ = std::thread(&_mxutil_stream_player_h::mxutil_stream_player_main_worker, this);
}

void print_ffmpeg_error_message(int errnum)
{
    char err_msg[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(errnum, err_msg, AV_ERROR_MAX_STRING_SIZE);
    printf("%s\n", err_msg);
}

void _mxutil_stream_player_h::mxutil_stream_player_reconnect()
{
    while (true)
    {
        if (avformat_open_input(&format_ctx_, stream_source_name.c_str(), NULL, &options_) != 0)
        {
            avformat_close_input(&format_ctx_);
            std::cout << "Failed to Re-open RTSP input stream: " << stream_source_name << "Retry in 1 seconds ..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        if (avformat_find_stream_info(format_ctx_, NULL) < 0)
        {
            std::cout << "Error: Could not find stream information." << std::endl;
            continue;
        }

        // stream demuxer
        for (unsigned int i = 0; i < format_ctx_->nb_streams; i++)
        {
            if (format_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
            {
                video_stream_index_ = i;
                break;
            }
        }

        if (video_stream_index_ == -1)
        {
            std::cout << "Error: Could not find a video stream." << std::endl;
            continue;
        }
        // play RTSP
        av_read_play(format_ctx_);
        break;
    }
}
void _mxutil_stream_player_h::mxutil_stream_player_main_worker()
{
    int ret;
    AVFrame* frame;

    while (running_)
    {
        // H.264 data in packet_
        if ((ret = av_read_frame(format_ctx_, &packet_)) < 0)
        {
            std::cerr << stream_source_name << ": av_read_frame failed: ";
            print_ffmpeg_error_message(ret);
            printf("%d %d\n", ret, AVERROR_EOF);
            if (ret == AVERROR_EOF)
            {
                avformat_close_input(&format_ctx_);
                mxutil_stream_player_reconnect();
            }
            continue;
        }

        if (packet_.stream_index != video_stream_index_)
        {
            av_packet_unref(&packet_);
            continue;
        }

        // Send packet to decoder
        ret = avcodec_send_packet(codec_ctx_, &packet_);
        if (ret == AVERROR(EAGAIN))
        {
            av_packet_unref(&packet_);
            continue;
        }
        else if (ret < 0)
        {
            std::cerr << stream_source_name << ": avcodec_send_packet failed: ";
            print_ffmpeg_error_message(ret);
            av_packet_unref(&packet_);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // Receive decoded frame
        while ((ret = avcodec_receive_frame(codec_ctx_, frame_yuv_)) >= 0)
        {
            // Check frame validity
            if (!frame_yuv_ || !frame_yuv_->data[0])
            {
                std::cerr << "[WARNING] Received invalid or empty frame.\n";
                av_frame_unref(frame_yuv_);
                continue;
            }

            if (frame_yuv_->flags & AV_FRAME_FLAG_CORRUPT)
            {
                std::cerr << "[WARNING] Decoder returned a corrupt frame. Skipping.\n";
                av_frame_unref(frame_yuv_);
                continue;
            }

            // Handle dynamic resolution changes
            if (frame_yuv_->width != codec_ctx_->width || frame_yuv_->height != codec_ctx_->height)
            {
                std::cerr << "[INFO] Resolution change detected. Reinitializing sws context.\n";
                sws_freeContext(img_convert_ctx_);
                img_convert_ctx_ = sws_getContext(
                    frame_yuv_->width, frame_yuv_->height, codec_ctx_->pix_fmt,
                    disp_width_, disp_height_, AV_PIX_FMT_RGB24,
                    SWS_BICUBIC, NULL, NULL, NULL);
            }

            // Convert YUV to RGB if buffer is available
            if (!available_frame_bufs_.empty())
            {
                frame = available_frame_bufs_.pop();

                if (!frame || !frame->data[0]) {
                    available_frame_bufs_.push(frame); // recycle
                    continue;
                } else {
                    int scale_ret = sws_scale(img_convert_ctx_,
                                        frame_yuv_->data, frame_yuv_->linesize,
                                        0, codec_ctx_->height,
                                        frame->data, frame->linesize);
                    if (scale_ret != frame->height) {
                        std::cerr << "[WARNING] Scale failed: " << scale_ret << "\n";
                        av_frame_unref(frame_yuv_);
                        available_frame_bufs_.push(frame);  // recycle
                        continue;
                    }
                }
                frames_.push(frame);
            }

            av_frame_unref(frame_yuv_);
        }

        if (ret != AVERROR(EAGAIN) && ret != AVERROR_EOF)
        {
            std::cerr << stream_source_name << ": avcodec_receive_frame failed: ";
            print_ffmpeg_error_message(ret);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        av_packet_unref(&packet_);
    }
}

std::string mxutil_stream_player_get_source_ip_addr(mxutil_stream_player_h stream_handle)
{
    _mxutil_stream_player_h *ctx = (_mxutil_stream_player_h *)stream_handle;

    std::string name = ctx->stream_source_name;

    std::size_t start = ctx->stream_source_name.find_first_of("@") + 1;
    std::size_t end = ctx->stream_source_name.find_last_of("/") - 1;
    std::size_t len = end - start + 1;

    return name.substr(start, len);
}

void mxutil_stream_get_input_resolution(mxutil_stream_player_h stream_handle, int &width, int &height)
{
    _mxutil_stream_player_h *ctx = (_mxutil_stream_player_h *)stream_handle;
    width = ctx->stream_frame_width_;
    height = ctx->stream_frame_height_;
}

void *mxutil_stream_player_get_frame(mxutil_stream_player_h stream_handle)
{
    _mxutil_stream_player_h *ctx = (_mxutil_stream_player_h *)stream_handle;

    AVFrame *frame = ctx->frames_.pop();

    ctx->buf = frame;

    return (void *)frame->data[0];
}

void mxutil_stream_player_return_buf(mxutil_stream_player_h stream_handle)
{
    _mxutil_stream_player_h *ctx = (_mxutil_stream_player_h *)stream_handle;

    ctx->available_frame_bufs_.push(ctx->buf);
}

void mxutil_stream_player_close(mxutil_stream_player_h stream_handle)
{
    _mxutil_stream_player_h *ctx = (_mxutil_stream_player_h *)stream_handle;

    ctx->running_ = false;

    delete ctx;
}

mxutil_stream_player_h mxutil_stream_player_open(const char *stream_url, const int disp_width_, const int disp_height_)
{
    try
    {
        _mxutil_stream_player_h *_stream_ctx = new _mxutil_stream_player_h(stream_url, disp_width_, disp_height_);
        return (mxutil_stream_player_h)_stream_ctx;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        exit(EXIT_FAILURE);
    }
}
