"""
============
Information:
============
Project: YOLOv7-tiny example code on MXA
File Name: yolov7.py

============
Description:
============
A script to show how to use the Acclerator API to perform a real-time inference
on MX3 using YOLOv7-tiny model.
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

class YoloV7Tiny:
    """
    A helper class to run YOLOv7 pre- and post-proccessing.
    """

###################################################################################################
    def __init__(self, stream_img_size=None):
        """
        The initialization function.
        """

        self.name = 'YoloV7Tiny-416'
        self.input_size = (416,416,3) 

        self.stream_mode = False
        if stream_img_size:
            # Pre-calculate ratio/pad values for preprocessing
            self.preprocess(np.zeros(stream_img_size))
            self.stream_mode = True

###################################################################################################
    def preprocess(self, img):
        """
        YOLOv7 Pre-proccessing.
        """
        h0, w0 = img.shape[:2] # orig hw

        r = self.input_size[0] / max(h0, w0)  # resize img to img_size
        if r != 1:  
            interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        h, w = img.shape[:2]

        img, ratio, dwdh = self._letterbox(img, new_shape=self.input_size, auto=False)
        img = img.astype(np.float32)
        img /= 255.0 # Scale

        shapes = (h0, w0), ((h / h0, w / w0), dwdh)

        if not self.stream_mode:
            self.ratio = r
            self.pad = dwdh
        img = np.expand_dims(img, 0)
        img = np.transpose(img, (0,3,1,2))
        return img
    
###################################################################################################
    def _letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=False, stride=32):
        """
        A letterbox function.
        """
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

        return im, r, (dw, dh)

###################################################################################################
    def postprocess(self, fmap):
        """
        YOLOv7 Post-proccessing.
        """
        
        post_output = fmap[0] # The output from the chip after postprocessing is done

        dets = []
        # run post process model
        for i, arr in enumerate(post_output):
            if arr[6] < 0.4:
                continue
            unpad = arr[1:5]-np.array([self.pad[0], self.pad[1], self.pad[0], self.pad[1]])
            x1,y1,x2,y2 = (unpad / self.ratio).astype(int)
            det = {}
            det['bbox'] = (x1,y1,x2,y2)
            det['class'] = COCO_CLASSES[int(arr[5])]
            det['class_idx'] = int(arr[5])
            det['score'] = arr[6]
            dets.append(det)
        return dets

###################################################################################################
if __name__=="__main__":
    pass

# eof