#pragma once

#include <opencv2/opencv.hpp>

#include "ipcam_stream.h"
#include "vdo_predec.h"

enum VideoInputType_e
{
    VIDEO_FROM_IPCAM, /* enum for ip camera*/
    VIDEO_FROM_FILE,  /* enum for video file*/
    VIDEO_FROM_USBCAM /* enum for usb camera*/
};

struct VideoInputSource_s
{
    VideoInputType_e type;
    std::string access_value;
};

/**
 * @brief Abstract class for streaming input sources.
 *
 * This class serves as a base for various streaming input sources, including IP cameras, USB cameras, and video files.
 */
class InputSource
{
public:
    virtual ~InputSource() {}
    virtual void GetFrame(cv::Mat &) {} /* get frame raw data from input source */
    virtual void ReturnFrame() {}       /* return frame buffer as needed */
    virtual void GetInputResolution(int & /* width */, int & /* height */) {}
};

class IpCamStream : public InputSource
{
private:
    mxutil_stream_player_h stream_ctx_;

public:
    /**
     * @brief Constructs a new IP camera stream object.
     *
     * This constructor initializes a stream from an IP camera using the specified URL and configures it with the desired display dimensions.
     *
     * @param stream_url The URL of the IP camera.
     * @param disp_width The desired width for display.
     * @param disp_height The desired height for display.
     */
    IpCamStream(const char *stream_url, const int disp_width, const int disp_height)
    {
        stream_ctx_ = mxutil_stream_player_open(stream_url, disp_width, disp_height);
    }

    // Destructor
    ~IpCamStream()
    {
        mxutil_stream_player_close(stream_ctx_);
    }

    /**
     * @brief Get frame raw data from ip camera
     */
    void GetFrame(cv::Mat &frame) override
    {
        if (frame.empty()) {
            std::cerr << "GetFrame: frame mat empty\n";
            return;
        }

        void* data = mxutil_stream_player_get_frame(stream_ctx_);
        if (!data) {
            std::cerr << "mxutil_stream_player_get_frame error: data null\n";
        } else {
            memcpy(frame.data, data, frame.total() * frame.elemSize());
        }

        this->ReturnFrame();
        return;
    }

    /**
     * @brief Return frame buffer, must be called following every GetFrame
     */
    void ReturnFrame() override
    {
        mxutil_stream_player_return_buf(stream_ctx_);
    }

    /**
     * @brief Get the ip camera streaming input resolution
     */
    void GetInputResolution(int &width, int &height) override
    {
        mxutil_stream_get_input_resolution(stream_ctx_, width, height);
    }

    /**
     * @brief Get the ip camera url
     */
    std::string GetIpAddress()
    {
        return mxutil_stream_player_get_source_ip_addr(stream_ctx_);
    }
};

class VideoFileStream : public InputSource
{
private:
    mxutil_vdo_player_h vfctx_;

public:
    /**
     * @brief Constructs a new VideoFileStream object.
     *
     * This constructor initializes a video file stream object that reads
     * from the specified file and prepares the video for display with the
     * given dimensions and frame rate.
     *
     * @param file_path The path to the video file.
     * @param disp_width The desired width for display.
     * @param disp_height The desired height for display.
     * @param num_predec_frames The number of frames to pre-decode for smoother playback.
     * @param target_fps The desired frames per second (FPS) for display.
     */
    VideoFileStream(const char *file_path, const int disp_width, const int disp_height, int num_predec_frames, int target_fps)
    {
        vfctx_ = mxutil_vdo_player_decode(file_path, num_predec_frames, disp_width, disp_height, FRAME_FMT_RGB, target_fps);
    }

    // Destructor
    ~VideoFileStream()
    {
        mxutil_vdo_player_close(vfctx_);
    }

    /**
     * @brief Get frame raw data from video stream
     */
    void GetFrame(cv::Mat &frame) override
    {
        void *data = mxutil_vdo_player_get_frame(vfctx_);
        memcpy(frame.data, data, frame.total() * frame.elemSize());

        this->ReturnFrame();
        return;
    }

    /**
     * @brief Return frame buffer as needed
     */
    void ReturnFrame() override
    {
    }

    /**
     * @brief Get the video file input resolution
     */
    void GetInputResolution(int &width, int &height) override
    {
        mxutil_vdo_player_get_frame_resolution(vfctx_, width, height);
    }
};

class VideoFileStreamReal : public InputSource
{
private:
    mxutil_vdo_player_real_h vfctx_;

public:
    /**
     * @brief Constructs a new VideoFileStreamReal object.
     *
     * This constructor initializes a video file stream object that reads
     * from the specified file and prepares the video for display with the
     * given dimensions and frame rate.
     *
     * @param file_path The path to the video file.
     * @param disp_width The desired width for display.
     * @param disp_height The desired height for display.
     * @param num_predec_frames The number of frames to pre-decode for smoother playback.
     * @param target_fps The desired frames per second (FPS) for display.
     */
    VideoFileStreamReal(const char *file_path, const int disp_width, const int disp_height)
    {
        vfctx_ = mxutil_vdo_player_real(file_path, disp_width, disp_height);
    }

    // Destructor
    ~VideoFileStreamReal()
    {
        mxutil_vdo_player_close_real(vfctx_);
    }

    /**
     * @brief Get frame raw data from video stream
     */
    void GetFrame(cv::Mat &frame) override
    {
        void *data = mxutil_vdo_player_get_frame_real(vfctx_);
        memcpy(frame.data, data, frame.total() * frame.elemSize());

        this->ReturnFrame();
        return;
    }

    /**
     * @brief Return frame buffer as needed
     */
    void ReturnFrame() override
    {
        return mxutil_vdo_player_return_frame_real(vfctx_);
    }

    /**
     * @brief Get the video file input resolution
     */
    void GetInputResolution(int & /*width */, int & /*height*/) override
    {
    }
};

class UsbCamStream : public InputSource
{
private:
    bool isOpened;
    cv::VideoCapture capture;

    cv::Mat img_mat;

    int m_cam_width;
    int m_cam_height;

    int m_resized_width;
    int m_resized_height;

public:
    /**
     * @brief Constructs a USB camera stream object.
     *
     * This constructor initializes a stream from a USB camera, configuring it with the specified display dimensions.
     *
     * @param dev_fd The file descriptor of the camera. Set to 0 if using /dev/video0, or the corresponding number for other devices.
     * @param disp_width The desired width for display.
     * @param disp_height The desired height for display.
     */
    UsbCamStream(int dev_fd, const int disp_width, const int disp_height)
    {
        capture.open(dev_fd, cv::CAP_V4L);
        if (!capture.isOpened())
        {
            printf("opencv open camera failed\n");
            isOpened = false;
            return;
        }

        isOpened = true;

        capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        capture.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        capture.set(cv::CAP_PROP_CONVERT_RGB, 0);

        m_cam_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
        m_cam_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);

        m_resized_width = disp_width;
        m_resized_height = disp_height;

        img_mat = cv::Mat(m_resized_height, m_resized_width, CV_8UC3);
    }

    // Destructor
    ~UsbCamStream()
    {
        if (isOpened)
            capture.release();
    }

    /**
     * @brief Get frame raw data from usb camera
     */
    void GetFrame(cv::Mat &frame) override
    {
        if (!isOpened) {
            printf("usbcam is not open\n");
            return;
        }

        cv::Mat jpg_mat;
        cv::Mat decoded_mat;

        if (false == capture.read(jpg_mat))
        {
            printf("opencv read video file failed\n");
            return;
        }

        // decoding JPEG

        cv::imdecode(jpg_mat, cv::IMREAD_COLOR, &decoded_mat);
        cv::cvtColor(decoded_mat, decoded_mat, cv::COLOR_BGR2RGB);

        // image resizing
        cv::resize(decoded_mat, frame, cv::Size(m_resized_width, m_resized_height), cv::INTER_LINEAR);

        this->ReturnFrame();
        return;
    }

    /**
     * @brief Return frame buffer as needed
     */
    void ReturnFrame() override
    {
    }

    /**
     * @brief Get the usb camera input resolution
     */
    void GetInputResolution(int &width, int &height) override
    {
        if (!isOpened)
            return;

        width = m_cam_width;
        height = m_cam_height;
    }
};
