# -*- coding: utf-8 -*-
# Copyright © 2021 Patrick Levin
# SPDX-Identifier: MIT
u"""BlazeFace face detection.

Ported from Google® MediaPipe (https://google.github.io/mediapipe/).

Model card:

    https://mediapipe.page.link/blazeface-mc

Reference:

    V. Bazarevsky et al. BlazeFace: Sub-millisecond
    Neural Face Detection on Mobile GPUs. CVPR
    Workshop on Computer Vision for Augmented and
    Virtual Reality, Long Beach, CA, USA, 2019.
"""
import numpy as np
import tensorflow as tf
from enum import IntEnum
import cv2 as cv
from PIL import Image
from typing import List, Optional, Tuple, Union
from fdlite.nms import non_maximum_suppression
from fdlite.transform import detection_letterbox_removal, bbox_to_roi, image_to_tensor
from fdlite.transform import sigmoid
from fdlite.types import Detection, Rect
from fdlite.transform import SizeMode

# score limit is 100 in mediapipe and leads to overflows with IEEE 754 floats
# this lower limit is safe for use with the sigmoid functions and float32
RAW_SCORE_LIMIT = 80
# threshold for confidence scores
MIN_SCORE = 0.5
# NMS similarity threshold
MIN_SUPPRESSION_THRESHOLD = 0.3

ROI_SCALE = (1.5, 1.5)          # Scaling of the face detection ROI

# from mediapipe module; irrelevant parts removed
# (reference: mediapipe/modules/face_detection/face_detection_short_cpu.pbtxt)
SSD_OPTIONS_SHORT = {
    'num_layers': 4,
    'input_size_height': 128,
    'input_size_width': 128,
    'anchor_offset_x': 0.5,
    'anchor_offset_y': 0.5,
    'strides': [8, 16, 16, 16],
    'interpolated_scale_aspect_ratio': 1.0
}

class FaceIndex(IntEnum):
    """Indexes of keypoints returned by the face detection model.

    Use these with detection results (by indexing the result):
    ```
        def get_left_eye_position(detection):
            x, y = detection[FaceIndex.LEFT_EYE]
            return x, y
    ```
    """
    LEFT_EYE = 0
    RIGHT_EYE = 1
    NOSE_TIP = 2
    MOUTH = 3
    LEFT_EYE_TRAGION = 4
    RIGHT_EYE_TRAGION = 5

class FaceDetectionMXA:
    def __init__(
        self
    ) -> None:
        ssd_opts = SSD_OPTIONS_SHORT
        self.input_shape = (1,128,128,3)
        self.anchors = _ssd_generate_anchors(ssd_opts)

    def _preprocess(self, frame, roi=None):
        height, width = self.input_shape[1:3]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        image_data = image_to_tensor(
            image,
            roi,
            output_size=(width, height),
            keep_aspect_ratio=True,
            output_range=(-1, 1))
        input_data = image_data.tensor_data[np.newaxis]
        #input_data = np.squeeze(input_data, axis=0)
        input_data = input_data.astype(np.float32)
        padding = image_data.padding
        return input_data, padding

    def _postprocess(self, outputs, padding, image_size):
        raw_boxes, raw_scores = self._get_raw_boxes_scores(outputs)
        detections, face_roi = self._get_detections_roi(raw_boxes, raw_scores, padding, image_size)
        return detections, face_roi

    def _get_raw_boxes_scores(self, outputs):
        classificator_8 = np.expand_dims(outputs[0], 0)
        classificator_16 = np.expand_dims(outputs[1], 0)
        regressor_8 = np.expand_dims(outputs[2], 0)
        regressor_16 = np.expand_dims(outputs[3], 0)
        
        # Run Post-Process Inference
        post_model_path = "../../models/model_0_blaze_face_short_range_post.tflite"
        self.interpreter = tf.lite.Interpreter(model_path=post_model_path)
        self.interpreter.allocate_tensors()
        
        classificator_8_index = self.interpreter.get_input_details()[0]['index']
        classificator_16_index = self.interpreter.get_input_details()[1]['index']
        regressor_8_index = self.interpreter.get_input_details()[2]['index']
        regressor_16_index = self.interpreter.get_input_details()[3]['index']
        
        classificators_index = self.interpreter.get_output_details()[0]['index']
        regressors_index = self.interpreter.get_output_details()[1]['index']
        self.interpreter.set_tensor(classificator_8_index, np.squeeze(classificator_8, axis=0))
        self.interpreter.set_tensor(classificator_16_index, np.squeeze(classificator_16, axis=0))
        self.interpreter.set_tensor(regressor_8_index, np.squeeze(regressor_8, axis=0))
        self.interpreter.set_tensor(regressor_16_index, np.squeeze(regressor_16, axis=0))
        self.interpreter.invoke()

        raw_scores = self.interpreter.get_tensor(classificators_index)
        raw_boxes = self.interpreter.get_tensor(regressors_index)
        return raw_boxes, raw_scores
    
    def _get_detections_roi(self, raw_boxes, raw_scores, padding, image_size):
        boxes = self._decode_boxes(raw_boxes)
        scores = self._get_sigmoid_scores(raw_scores)
        detections = self._convert_to_detections(boxes, scores)
        pruned_detections = non_maximum_suppression(
                                detections,
                                MIN_SUPPRESSION_THRESHOLD, MIN_SCORE,
                                weighted=True)
        detections = detection_letterbox_removal(pruned_detections, padding)
        
        # get ROI for the first face found
        if len(detections) > 0:
            face_roi = face_detection_to_roi(detections[0], image_size)
            return detections, face_roi
        else: return detections, None
    
    def _decode_boxes(self, raw_boxes: np.ndarray) -> np.ndarray:
        """Simplified version of
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
        """
        # width == height so scale is the same across the board
        scale = self.input_shape[1]
        num_points = raw_boxes.shape[-1] // 2
        # scale all values (applies to positions, width, and height alike)
        boxes = raw_boxes.reshape(-1, num_points, 2) / scale
        # adjust center coordinates and key points to anchor positions
        boxes[:, 0] += self.anchors
        for i in range(2, num_points):
            boxes[:, i] += self.anchors
        # convert x_center, y_center, w, h to xmin, ymin, xmax, ymax
        center = np.array(boxes[:, 0])
        half_size = boxes[:, 1] / 2
        boxes[:, 0] = center - half_size
        boxes[:, 1] = center + half_size
        return boxes

    def _get_sigmoid_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        """Extracted loop from ProcessCPU (line 327) in
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
        """
        # just a single class ("face"), which simplifies this a lot
        # 1) thresholding; adjusted from 100 to 80, since sigmoid of [-]100
        #    causes overflow with IEEE single precision floats (max ~10e38)
        raw_scores[raw_scores < -RAW_SCORE_LIMIT] = -RAW_SCORE_LIMIT
        raw_scores[raw_scores > RAW_SCORE_LIMIT] = RAW_SCORE_LIMIT
        # 2) apply sigmoid function on clipped confidence scores
        return sigmoid(raw_scores)

    @staticmethod
    def _convert_to_detections(
        boxes: np.ndarray,
        scores: np.ndarray
    ) -> List[Detection]:
        """Apply detection threshold, filter invalid boxes and return
        detection instance.
        """
        # return whether width and height are positive
        def is_valid(box: np.ndarray) -> bool:
            return bool(np.all(box[1] > box[0]))

        score_above_threshold = scores > MIN_SCORE
        filtered_boxes = boxes[np.argwhere(score_above_threshold)[:, 1], :]
        filtered_scores = scores[score_above_threshold]
        return [Detection(box, score)
                for box, score in zip(filtered_boxes, filtered_scores)
                if is_valid(box)]

def _ssd_generate_anchors(opts: dict) -> np.ndarray:
    """This is a trimmed down version of the C++ code; all irrelevant parts
    have been removed.
    (reference: mediapipe/calculators/tflite/ssd_anchors_calculator.cc)
    """
    layer_id = 0
    num_layers = opts['num_layers']
    strides = opts['strides']
    assert len(strides) == num_layers
    input_height = opts['input_size_height']
    input_width = opts['input_size_width']
    anchor_offset_x = opts['anchor_offset_x']
    anchor_offset_y = opts['anchor_offset_y']
    interpolated_scale_aspect_ratio = opts['interpolated_scale_aspect_ratio']
    anchors = []
    while layer_id < num_layers:
        last_same_stride_layer = layer_id
        repeats = 0
        while (last_same_stride_layer < num_layers and
               strides[last_same_stride_layer] == strides[layer_id]):
            last_same_stride_layer += 1
            # aspect_ratios are added twice per iteration
            repeats += 2 if interpolated_scale_aspect_ratio == 1.0 else 1
        stride = strides[layer_id]
        feature_map_height = input_height // stride
        feature_map_width = input_width // stride
        for y in range(feature_map_height):
            y_center = (y + anchor_offset_y) / feature_map_height
            for x in range(feature_map_width):
                x_center = (x + anchor_offset_x) / feature_map_width
                for _ in range(repeats):
                    anchors.append((x_center, y_center))
        layer_id = last_same_stride_layer
    return np.array(anchors, dtype=np.float32)

def face_detection_to_roi(
    face_detection: Detection,
    image_size: Tuple[int, int]
) -> Rect:
    """Return a normalized ROI from a list of face detection results.

    The result of this function is intended to serve as the input of
    calls to `FaceLandmark`:

    ```
        MODEL_PATH = '/var/mediapipe/models/'
        ...
        face_detect = FaceDetection(model_path=MODEL_PATH)
        face_landmarks = FaceLandmark(model_path=MODEL_PATH)
        image = Image.open('/home/user/pictures/photo.jpg')
        # detect faces
        detections = face_detect(image)
        for detection in detections:
            # find ROI from detection
            roi = face_detection_to_roi(detection)
            # extract face landmarks using ROI
            landmarks = face_landmarks(image, roi)
            ...
    ```

    Args:
        face_detection (Detection): Normalized face detection result from a
            call to `FaceDetection`.

        image_size (tuple): A tuple of `(image_width, image_height)` denoting
            the size of the input image the face detection results came from.

    Returns:
        (Rect) Normalized ROI for passing to `FaceLandmark`.
    """
    absolute_detection = face_detection.scaled(image_size)
    left_eye = absolute_detection[FaceIndex.LEFT_EYE]
    right_eye = absolute_detection[FaceIndex.RIGHT_EYE]
    return bbox_to_roi(
        face_detection.bbox,
        image_size,
        rotation_keypoints=[left_eye, right_eye],
        scale=ROI_SCALE,
        size_mode=SizeMode.SQUARE_LONG
    )

