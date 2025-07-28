"""
============
Information:
============
Project: YOLOv8 example code on MXA
File Name: yolov8.py

============
Description:
============
A script to show how to use the Acclerator API to perform a real-time inference
on MX3 using YOLOv8 model.
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

class YoloV8:
    """
    A helper class to run YOLOv8 pre- and post-proccessing.
    """

###################################################################################################
    def __init__(self, stream_img_size=None, model_type='tflite'):
        """
        The initialization function.
        """

        self.name = 'YoloV8'
        self.input_size = (640,640,3) 
        self.input_width = 640
        self.input_height = 640
        self.confidence_thres = 0.4
        self.iou_thres = 0.6
        self.model_type = model_type

        self.stream_mode = False
        if stream_img_size:
            # Pre-calculate ratio/pad values for preprocessing
            self.preprocess(np.zeros(stream_img_size))
            self.stream_mode = True

###################################################################################################
    def preprocess(self, img):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        self.original_imgage = img

        # Get the height and width of the input image
        [self.img_height, self.img_width, _] = self.original_imgage.shape
        
        # Prepare a square image for inference
        self.length = max((self.img_height, self.img_width))
        self.image = np.zeros((self.length, self.length, 3), np.uint8)
        self.image[0:self.img_height, 0:self.img_width] = self.original_imgage

        # Calculate scale factor
        scale = self.length / 640

        # Preprocess the image and prepare blob for model
        blob = cv2.dnn.blobFromImage(self.image, scalefactor=1 / 255, size=(640, 640), swapRB=True)

        if self.model_type == 'tflite':
            # Assume 'blob' is currently (1, 3, 640, 640)
            blob = blob.transpose(0, 2, 3, 1)  # Change to (1, 640, 640, 3)

        # Return the preprocessed image data
        return blob

###################################################################################################
    def postprocess(self, output):
        """
        Performs post-processing on the YOLOv8 model's output to extract bounding boxes, scores, and class IDs.

        Args:
            output (numpy.ndarray): The output of the model.

        Returns:
            list: A list of detections where each detection is a dictionary containing 
                    'bbox', 'class_id', 'class', and 'score'.
        """
        # Transpose the output to shape (8400, 84)
        outputs = np.transpose(np.squeeze(output[0]))

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.length / self.input_width
        y_factor = self.length / self.input_height

        # Extract the bounding box information and class scores in a vectorized manner
        boxes = outputs[:, :4]  # (8400, 4) - x_center, y_center, width, height
        class_scores = outputs[:, 4:]  # (8400, 80) - class scores for 80 classes

        # Find the class with the highest score for each detection
        max_scores = np.max(class_scores, axis=1)  # (8400,) - maximum class score for each detection
        class_ids = np.argmax(class_scores, axis=1)  # (8400,) - index of the best class

        # Filter out detections with scores below the confidence threshold
        valid_indices = np.where(max_scores >= self.confidence_thres)[0]
        if len(valid_indices) == 0:
            return []  # Return an empty list if no valid detections

        # Select only valid detections
        valid_boxes = boxes[valid_indices]
        valid_class_ids = class_ids[valid_indices]
        valid_scores = max_scores[valid_indices]

        # Convert bounding box coordinates from (x_center, y_center, w, h) to (left, top, width, height)
        valid_boxes[:, 0] = (valid_boxes[:, 0] - valid_boxes[:, 2] / 2) * x_factor  # left
        valid_boxes[:, 1] = (valid_boxes[:, 1] - valid_boxes[:, 3] / 2) * y_factor  # top
        valid_boxes[:, 2] = valid_boxes[:, 2] * x_factor  # width
        valid_boxes[:, 3] = valid_boxes[:, 3] * y_factor  # height

        # Create detection dictionaries
        detections = [{
            'bbox': valid_boxes[i].astype(int).tolist(),
            'class_id': int(valid_class_ids[i]),
            'class': COCO_CLASSES[int(valid_class_ids[i])],
            'score': valid_scores[i]
        } for i in range(len(valid_indices))]

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        if len(detections) > 0:
            # NMS requires two lists: bounding boxes and confidence scores
            boxes_for_nms = [d['bbox'] for d in detections]
            scores_for_nms = [d['score'] for d in detections]

            indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores_for_nms, self.confidence_thres, self.iou_thres)

            # Check if indices is not empty
            if len(indices) > 0:
                # Flatten indices if they are returned as a list of arrays
                if isinstance(indices[0], list) or isinstance(indices[0], np.ndarray):
                    indices = [i[0] for i in indices]

                # Filter detections based on NMS
                final_detections = [detections[i] for i in indices]
            else:
                final_detections = []
        else:
            final_detections = []

        # Return the list of final detections
        return final_detections


###################################################################################################
if __name__=="__main__":
    pass

# eof
