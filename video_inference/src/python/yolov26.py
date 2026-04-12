"""
============
Information:
============
Project: YOLOv26 example code on MXA
File Name: yolov26.py

============
Description:
============
A script to show how to use the Acclerator API to perform a real-time inference
on MX3 using YOLOv26 model.
"""

###################################################################################################

# Imports

import numpy as np
import cv2

###################################################################################################

COCO_CLASSES = ( "person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
        "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",)

###################################################################################################
###################################################################################################
###################################################################################################

class YoloV26:
    """
    A helper class to run YOLOv26 pre- and post-proccessing.
    """

###################################################################################################
    def __init__(self, stream_img_size=None):
        """
        The initialization function.
        """

        self.name = 'YoloV26-640'
        self.input_size = (640, 640, 3) 

        self.stream_mode = False
        if stream_img_size:
            # Pre-calculate ratio/pad values for preprocessing
            self.preprocess(np.zeros(stream_img_size))
            self.stream_mode = True


    def preprocess(self, img):
        """
        YOLOv26 Pre-processing.
        """
        h0, w0 = img.shape[:2]
        self.orig_shape = (h0, w0)

        r = self.input_size[0] / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)

        img, _, dwdh = self._letterbox(img, new_shape=self.input_size, auto=False)

        img = img.astype(np.float32) / 255.0

        # Always store these because postprocess needs them
        self.ratio = r
        self.pad = dwdh

        # HWC -> BCHW
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img


    def _letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=False, stride=32):
        """
        Letterbox image to target shape.
        """
        shape = im.shape[:2]  # (h, w)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)

        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]

        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return im, r, (dw, dh)

    def postprocess(self, fmap):

        out = np.array(fmap).squeeze()

        if out.shape[0] == 6:
            out = out.T
        else:
            out = out.reshape(-1, 6)

        boxes = []
        scores = []

        for d in out:
            x, y, w, h, obj, cls = d

            conf = float(obj)
            if conf < 0.5:
                continue

            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2

            boxes.append([x1, y1, x2, y2])
            scores.append(conf)

        # -----------------------
        # NMS (IMPORTANT FIX)
        # -----------------------
        idxs = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            score_threshold=0.5,
            nms_threshold=0.4
        )

        dets = []

        if len(idxs) > 0:
            for i in idxs.flatten():
                x1, y1, x2, y2 = boxes[i]

                dets.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "score": float(scores[i]),
                    "class_idx": 0
                })

        return dets

###################################################################################################
if __name__=="__main__":
    pass

# eof
