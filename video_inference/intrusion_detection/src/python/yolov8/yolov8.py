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
    def __init__(self, stream_img_size=None):
        """
        The initialization function.
        """

        self.name = 'YoloV8'
        self.input_size = (640,640,3) 
        self.input_width = 640
        self.input_height = 640
        self.confidence_thres = 0.001
        self.iou_thres = 0.6

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
        self.img = img

        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]
        
        img = cv2.convertScaleAbs(img)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_width, self.input_height))
        
        img = img.astype(np.float32)
        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0


        # Expand dimensions to add the batch size
        image_data = np.expand_dims(image_data, axis=0)  

        # Return the preprocessed image data
        return image_data

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
        x_factor = self.img_width
        y_factor = self.img_height

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
