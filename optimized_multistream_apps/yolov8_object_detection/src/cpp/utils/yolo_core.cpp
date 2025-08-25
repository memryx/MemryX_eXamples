#include "yolo_core.h"

#define FONT (cv::FONT_ITALIC)

const char *COCO_NAMES[COCO_CLASS_NUMBER] = {
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
};

float intersection_over_union(BBox &bbox_0, BBox &bbox_1, int class_chk)
{
    if (class_chk)
    {
        if (bbox_0.class_index != bbox_1.class_index)
            return 0.0;
    }

    float y_min = mxutil_max(bbox_0.y_min, bbox_1.y_min);
    float x_min = mxutil_max(bbox_0.x_min, bbox_1.x_min);
    float y_max = mxutil_min(bbox_0.y_max, bbox_1.y_max);
    float x_max = mxutil_min(bbox_0.x_max, bbox_1.x_max);
    float intersection_area = mxutil_max(0, (y_max - y_min)) * mxutil_max(0, (x_max - x_min));
    float bbox_0_area = (bbox_0.y_max - bbox_0.y_min) * (bbox_0.x_max - bbox_0.x_min);
    float bbox_1_area = (bbox_1.y_max - bbox_1.y_min) * (bbox_1.x_max - bbox_1.x_min);
    float union_area = bbox_0_area + bbox_1_area - intersection_area;
    return intersection_area / union_area;
}

void non_maximum_suppression(std::queue<BBox> &bboxes, BBox &bbox, float iou)
{
    BBox bbox_0;
    BBox bbox_to_be_stored;
    int count = bboxes.size();

    int exit_flag = 0;

    // iterative over all bounding boxes
    for (int i = 0; i < count; ++i)
    {
        bbox_0 = bboxes.front();
        bboxes.pop();
        if (intersection_over_union(bbox_0, bbox, 0) > iou)
        {
            // if two bounding boxes are highly overlapped, keep the one with higher score
            if (bbox_0.class_score > bbox.class_score)
            {
                bbox_to_be_stored = bbox_0;
                exit_flag = 1;
            }
            else
            {
                bbox_to_be_stored = bbox;
            }
            if (exit_flag)
            {
                bboxes.push(bbox_to_be_stored);
                return;
            }
            else
            {
                if (i == count - 1)
                { // end of comparison
                    bboxes.push(bbox_to_be_stored);
                    return;
                }
            }
        }
        else
        {
            // otherwise, put it back to list
            bboxes.push(bbox_0);
        }
    }
    // if not return from loop, we then add the given bounding box to list
    bboxes.push(bbox);
}

void _draw_bbox(cv::Mat &image, int x_min, int y_min, int x_max, int y_max,
                cv::Scalar box_color, cv::Scalar text_color, const char *class_name)
{

    double font_scale = ((double)image.rows / 640.0);
    double bbox_thickness = font_scale * 3;
    double font_thickness = font_scale * 2;
    cv::Size text_size;
    int baseline;
    char text[64];
    /* bounding box rectangle line */
    cv::rectangle(image, cv::Point(x_min, y_min) /*top left*/, cv::Point(x_max, y_max) /*bottom right*/,
                  box_color, bbox_thickness, cv::LINE_4);

    sprintf(text, "%s", class_name);

    text_size = cv::getTextSize(text, FONT, 2 * font_scale, bbox_thickness, &baseline);

    /* label background rectangle */
    cv::rectangle(image,
                  cv::Rect(x_min, mxutil_max(0, y_min - text_size.height), text_size.width * 0.5, text_size.height), // top left, width, height
                  box_color, cv::FILLED);

    /* label text */
    cv::putText(image, text,
                cv::Point(x_min, mxutil_max(0, y_min - text_size.height) == 0 ? text_size.height - 5 : y_min - 10 * font_scale), // bottom left
                FONT, font_scale, text_color, font_thickness, cv::LINE_AA);
}

void _draw_bbox(cv::Mat &image, int x_min, int y_min, int x_max, int y_max,
                cv::Scalar box_color, cv::Scalar text_color, const char *class_name, const float class_score)
{

    double font_scale = ((double)image.rows / 640.0);
    double bbox_thickness = font_scale * 3;
    double font_thickness = font_scale * 2;
    cv::Size text_size;
    int baseline;
    char text[64];
    /* bounding box rectangle line */
    cv::rectangle(image, cv::Point(x_min, y_min) /*top left*/, cv::Point(x_max, y_max) /*bottom right*/,
                  box_color, bbox_thickness, cv::LINE_4);

    sprintf(text, "%s(%.f%%)", class_name, 100 * class_score);

    text_size = cv::getTextSize(text, FONT, 2 * font_scale, bbox_thickness, &baseline);

    /* label background rectangle */
    cv::rectangle(image,
                  cv::Rect(x_min, mxutil_max(0, y_min - text_size.height), text_size.width * 0.5, text_size.height), // top left, width, height
                  box_color, cv::FILLED);

    /* label text */
    cv::putText(image, text,
                cv::Point(x_min, mxutil_max(0, y_min - text_size.height) == 0 ? text_size.height - 5 : y_min - 10 * font_scale), // bottom left
                FONT, font_scale, text_color, font_thickness, cv::LINE_AA);
}
