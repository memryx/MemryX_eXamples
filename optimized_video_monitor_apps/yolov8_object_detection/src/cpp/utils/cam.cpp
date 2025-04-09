#include "cam.h"

#include <map>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

#include <endian.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>

#include <fcntl.h> /* low-level i/o */
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include <linux/videodev2.h>

#define NUM_MMAP_BUFFER 4

struct mmap_buf_t
{
    void *start[3];
    size_t length[3];
};

typedef struct
{
    int fd;
    char dev_name[20];
    enum v4l2_buf_type vdo_buf_type;
    bool is_mplane;
    struct v4l2_plane *vdo_planes;
    int vdo_num_planes;
    struct mmap_buf_t *buffers;
    unsigned int n_buffers;

    std::map<void *, struct v4l2_buffer> v4l2_buf_map;

    int width;
    int height;
    int pixelformat;
} mxutil_camera_capture_body_t;

#define CLEAR_STRUCT(x) memset(&(x), 0, sizeof(x))

static int xioctl(int fh, int request, void *arg)
{
    int r;

    do
    {
        r = ioctl(fh, request, arg);
    } while (-1 == r && EINTR == errno);

    return r;
}

std::vector<int> mxutil_cam_filter_supported()
{
    std::vector<int> vec_camsup;
    vec_camsup.clear();

    // 64 is big enough .. FIXME
    for (int i = 0; i < 64; i++)
    {
        char vdo_path[64];
        sprintf(vdo_path, "/dev/video%d", i);
        int fd = open(vdo_path, O_RDWR);

        if (fd == -1)
        {
            // printf("%s open error\n", vdo_path);
            continue;
        }

        struct v4l2_capability cap;
        int ret = xioctl(fd, VIDIOC_QUERYCAP, &cap);

        if (ret == -1)
        {
            // printf("%s VIDIOC_QUERYCAP error\n", vdo_path);
            close(fd);
            continue;
        }

        struct v4l2_format fmt;
        CLEAR_STRUCT(fmt);

        if (cap.device_caps & V4L2_CAP_VIDEO_CAPTURE)
            fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        else if (cap.device_caps & V4L2_CAP_VIDEO_CAPTURE_MPLANE)
            fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        else
        {
            // printf("%s device_caps error : 0x%x\n", vdo_path, cap.device_caps);
            close(fd);
            continue;
        }

        ret = xioctl(fd, VIDIOC_G_FMT, &fmt);

        if (ret == -1)
        {
            // printf("%s VIDIOC_G_FMT error\n", vdo_path);
            close(fd);
            continue;
        }

        if (fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_MJPEG ||
            fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_YUYV)
        {
            // supported
            // printf("%s OK\n", vdo_path);
            vec_camsup.push_back(i);
        }

        close(fd);
    }

    return vec_camsup;
}

mxutil_cam_t mxutil_cam_open(int cam_id)
{
    mxutil_camera_capture_body_t *cc_bdy = (mxutil_camera_capture_body_t *)malloc(sizeof(mxutil_camera_capture_body_t));

    sprintf(cc_bdy->dev_name, "/dev/video%d", cam_id);

    struct stat st;

    if (-1 == stat(cc_bdy->dev_name, &st))
    {
        fprintf(stderr, "Cannot identify '%s': %d, %s\n",
                cc_bdy->dev_name, errno, strerror(errno));
        free(cc_bdy);
        return NULL;
    }

    if (!S_ISCHR(st.st_mode))
    {
        fprintf(stderr, "%s is no devicen", cc_bdy->dev_name);
        free(cc_bdy);
        return NULL;
    }

    int fd = open(cc_bdy->dev_name, O_RDWR /* required */ | O_NONBLOCK, 0);

    if (-1 == fd)
    {
        fprintf(stderr, "Cannot open '%s': %d, %s\n",
                cc_bdy->dev_name, errno, strerror(errno));
        free(cc_bdy);
        return NULL;
    }

    cc_bdy->fd = fd;

    struct v4l2_capability cap;
    struct v4l2_cropcap cropcap;
    struct v4l2_crop crop;
    struct v4l2_format fmt;
    unsigned int min;

    if (-1 == xioctl(fd, VIDIOC_QUERYCAP, &cap))
    {
        if (EINVAL == errno)
        {
            fprintf(stderr, "%s is no V4L2 device\n",
                    cc_bdy->dev_name);
            free(cc_bdy);
            return NULL;
        }
        else
        {
            printf("VIDIOC_QUERYCAP error\n");
            free(cc_bdy);
            return NULL;
        }
    }

    // printf("capabilities = 0x%x\n", cap.capabilities);

    if (!(cap.capabilities & (V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_VIDEO_CAPTURE_MPLANE)))
    {
        fprintf(stderr, "%s is no video capture device\n",
                cc_bdy->dev_name);
        free(cc_bdy);
        return NULL;
    }

    if (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)
    {
        cc_bdy->vdo_buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        cc_bdy->is_mplane = false;
    }
    else
    {
        cc_bdy->vdo_buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        cc_bdy->is_mplane = true;
    }

    if (!(cap.capabilities & V4L2_CAP_STREAMING))
    {
        fprintf(stderr, "%s does not support streaming i/o\n",
                cc_bdy->dev_name);
        free(cc_bdy);
        return NULL;
    }

    /* Select video input, video standard and tune here. */

    CLEAR_STRUCT(cropcap);

    cropcap.type = cc_bdy->vdo_buf_type;

    if (0 == xioctl(fd, VIDIOC_CROPCAP, &cropcap))
    {
        crop.type = cc_bdy->vdo_buf_type;
        crop.c = cropcap.defrect; /* reset to default */

        if (-1 == xioctl(fd, VIDIOC_S_CROP, &crop))
        {
            switch (errno)
            {
            case EINVAL:
                /* Cropping not supported. */
                break;
            default:
                /* Errors ignored. */
                break;
            }
        }
    }
    else
    {
        /* Errors ignored. */
    }

    CLEAR_STRUCT(fmt);

    fmt.type = cc_bdy->vdo_buf_type;
    if (-1 == xioctl(fd, VIDIOC_G_FMT, &fmt))
    {
        printf("VIDIOC_G_FMT error\n");
        free(cc_bdy);
        return NULL;
    }


    if (cc_bdy->is_mplane)
    {
        cc_bdy->vdo_num_planes = fmt.fmt.pix_mp.num_planes;
        cc_bdy->vdo_planes = (struct v4l2_plane *)calloc(cc_bdy->vdo_num_planes, sizeof(struct v4l2_plane));
    }

    cc_bdy->width = fmt.fmt.pix.width;
    cc_bdy->height = fmt.fmt.pix.height;
    cc_bdy->pixelformat = fmt.fmt.pix.pixelformat;

    /* Buggy driver paranoia. */
    min = fmt.fmt.pix.width * 2;
    if (fmt.fmt.pix.bytesperline < min)
        fmt.fmt.pix.bytesperline = min;
    min = fmt.fmt.pix.bytesperline * fmt.fmt.pix.height;
    if (fmt.fmt.pix.sizeimage < min)
        fmt.fmt.pix.sizeimage = min;

    // init mmap

    struct v4l2_requestbuffers req;

    CLEAR_STRUCT(req);

    req.count = NUM_MMAP_BUFFER;
    req.type = cc_bdy->vdo_buf_type;
    req.memory = V4L2_MEMORY_MMAP;

    if (-1 == xioctl(fd, VIDIOC_REQBUFS, &req))
    {
        if (EINVAL == errno)
        {
            fprintf(stderr, "%s does not support memory mapping", cc_bdy->dev_name);
            free(cc_bdy);
            return NULL;
        }
        else
        {
            printf("VIDIOC_REQBUFS error\n");
            free(cc_bdy);
            return NULL;
        }
    }

    if (req.count < 2)
    {
        fprintf(stderr, "Insufficient buffer memory on %s\n", cc_bdy->dev_name);
        free(cc_bdy);
        return NULL;
    }

    cc_bdy->buffers = (struct mmap_buf_t *)calloc(req.count, sizeof(struct mmap_buf_t));

    if (!cc_bdy->buffers)
    {
        fprintf(stderr, "Out of memory\n");
        free(cc_bdy);
        return NULL;
    }

    cc_bdy->v4l2_buf_map.clear();

    uint32_t nb = 0;
    for (nb = 0; nb < req.count; ++nb)
    {
        struct v4l2_buffer v4l2buf;

        CLEAR_STRUCT(v4l2buf);

        v4l2buf.type = cc_bdy->vdo_buf_type;
        v4l2buf.memory = V4L2_MEMORY_MMAP;
        v4l2buf.index = nb;

        if (cc_bdy->is_mplane)
        {
            v4l2buf.length = cc_bdy->vdo_num_planes;
            v4l2buf.m.planes = cc_bdy->vdo_planes;
            memset(cc_bdy->vdo_planes, 0, sizeof(struct v4l2_plane));
        }

        if (-1 == xioctl(fd, VIDIOC_QUERYBUF, &v4l2buf))
        {
            printf("VIDIOC_QUERYBUF error\n");
            free(cc_bdy);
            return NULL;
        }

        if (cc_bdy->is_mplane)
        {
            for (int i = 0; i < cc_bdy->vdo_num_planes; i++)
            {
                cc_bdy->buffers[nb].length[i] = v4l2buf.m.planes[i].length;
                cc_bdy->buffers[nb].start[i] = mmap(NULL, v4l2buf.m.planes[i].length,
                                                    PROT_READ | PROT_WRITE,
                                                    MAP_SHARED, fd,
                                                    v4l2buf.m.planes[i].m.mem_offset);
            }
        }
        else
        {
            cc_bdy->buffers[nb].length[0] = v4l2buf.length;
            cc_bdy->buffers[nb].start[0] = mmap(NULL /* start anywhere */,
                                                v4l2buf.length,
                                                PROT_READ | PROT_WRITE /* required */,
                                                MAP_SHARED /* recommended */,
                                                fd, v4l2buf.m.offset);
        }

        if (MAP_FAILED == cc_bdy->buffers[nb].start)
        {
            printf("mmap error\n");
            free(cc_bdy);
            return NULL;
        }

        if (-1 == xioctl(fd, VIDIOC_QBUF, &v4l2buf))
        {
            printf("VIDIOC_QBUF error\n");
            free(cc_bdy);
            return NULL;
        }

        void *frame_buf = cc_bdy->buffers[nb].start[0]; // FIXME for MP
        cc_bdy->v4l2_buf_map[frame_buf] = v4l2buf;
    }

    cc_bdy->n_buffers = nb;

    if (-1 == xioctl(fd, VIDIOC_STREAMON, &cc_bdy->vdo_buf_type))
    {
        printf("VIDIOC_STREAMON error\n");
        free(cc_bdy);
        return NULL;
    }

    return (mxutil_cam_t)cc_bdy;
}

// get camera settings
int mxutil_cam_get_setting(mxutil_cam_t cc, mxutil_cam_setting_t *cc_setting)
{
    mxutil_camera_capture_body_t *cc_bdy = (mxutil_camera_capture_body_t *)cc;

    cc_setting->width = cc_bdy->width;
    cc_setting->height = cc_bdy->height;

    switch (cc_bdy->pixelformat)
    {
    case V4L2_PIX_FMT_MJPEG:
        cc_setting->pixfmt = mxutil_IMG_FMT_MJPG;
        break;
    case V4L2_PIX_FMT_RGB24:
        cc_setting->pixfmt = mxutil_IMG_FMT_RGB24;
        break;
    case V4L2_PIX_FMT_GREY:
        cc_setting->pixfmt = mxutil_IMG_FMT_GREY;
        break;
    case V4L2_PIX_FMT_YUYV:
        cc_setting->pixfmt = mxutil_IMG_FMT_YUYV;
        break;
    default:
        cc_setting->pixfmt = mxutil_IMG_FMT_OTHERS;
        break;
    }

    return 0;
}

// read frame buffer pointer
void *mxutil_cam_get_frame(mxutil_cam_t cc)
{
    mxutil_camera_capture_body_t *cc_bdy = (mxutil_camera_capture_body_t *)cc;

    struct v4l2_buffer v4l2buf;

    for (;;)
    {
        fd_set fds;
        struct timeval tv;
        int r;

        CLEAR_STRUCT(v4l2buf);
        v4l2buf.type = cc_bdy->vdo_buf_type;
        v4l2buf.memory = V4L2_MEMORY_MMAP;

        FD_ZERO(&fds);
        FD_SET(cc_bdy->fd, &fds);

        /* Timeout. */
        tv.tv_sec = 5;
        tv.tv_usec = 0;

        r = select(cc_bdy->fd + 1, &fds, NULL, NULL, &tv);

        if (-1 == r)
        {
            if (EINTR == errno)
                continue;
            printf("v4l2 select error\n");
            return NULL;
        }

        if (0 == r)
        {
            fprintf(stderr, "v4l2 select timeout\n");
            exit(EXIT_FAILURE);
        }

        if (cc_bdy->is_mplane)
        {
            v4l2buf.length = cc_bdy->vdo_num_planes;
            v4l2buf.m.planes = cc_bdy->vdo_planes;
            memset(cc_bdy->vdo_planes, 0, sizeof(struct v4l2_plane));
        }

        if (-1 == xioctl(cc_bdy->fd, VIDIOC_DQBUF, &v4l2buf))
        {
            switch (errno)
            {
            case EAGAIN:
                continue;

            case EIO:
                /* Could ignore EIO, see spec. */
                /* fall through */

            default:
                printf("VIDIOC_DQBUF error: %s (errno: %d)\n", strerror(errno), errno);
                return NULL;
            }
        }

        break;
    }

    return cc_bdy->buffers[v4l2buf.index].start[0];
}

// return frame buffer pointer
int mxutil_cam_put_frame(mxutil_cam_t cc, void *frame_buf)
{
    mxutil_camera_capture_body_t *cc_bdy = (mxutil_camera_capture_body_t *)cc;

    struct v4l2_buffer &v4l2buf = cc_bdy->v4l2_buf_map[frame_buf];

    if (-1 == xioctl(cc_bdy->fd, VIDIOC_QBUF, &v4l2buf))
    {
        printf("VIDIOC_QBUF error\n");
        return -1;
    }

    return 0;
}

// close the camera
int mxutil_cam_close(mxutil_cam_t cc)
{
    mxutil_camera_capture_body_t *cc_bdy = (mxutil_camera_capture_body_t *)cc;

    // FIXME: free memory

    int ret = close(cc_bdy->fd);
    if (ret != 0)
    {
        printf("close error: %s (errno: %d)\n", strerror(errno), errno);
    }

    return ret;
}
