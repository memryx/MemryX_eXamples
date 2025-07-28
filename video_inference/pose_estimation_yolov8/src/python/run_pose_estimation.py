# OpenCV and helper libraries imports
import sys
import os
from pathlib import Path
from queue import Queue
import cv2 as cv
import numpy as np
from memryx import AsyncAccl
import torchvision.ops as ops
from typing import List
import argparse

class App:
    def __init__(self, cam, model_input_shape, mirror=False, src_is_cam=True, **kwargs):
        # Initialize camera and various configurations
        self.cam = cam
        self.input_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.input_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
        self.model_input_shape = model_input_shape
        self.capture_queue = Queue(maxsize=5)  # Queue to store frames for processing
        self.mirror = mirror  # Flag to mirror the video frame
        self.box_score = 0.25  # Threshold for object confidence
        self.ratio = None
        self.kpt_score = 0.5  # Threshold for keypoint confidence
        self.nms_thr = 0.2  # IoU threshold for non-max suppression
        self.src_is_cam = src_is_cam

        # Predefined color list for drawing keypoints
        self.COLOR_LIST = list([[128, 255, 0], [255, 128, 50], [128, 0, 255], [255, 255, 0],
                   [255, 102, 255], [255, 51, 255], [51, 153, 255], [255, 153, 153],
                   [255, 51, 51], [153, 255, 153], [51, 255, 51], [0, 255, 0],
                   [255, 0, 51], [153, 0, 153], [51, 0, 51], [0, 0, 0],
                   [0, 102, 255], [0, 51, 255], [0, 153, 255], [0, 153, 153]])

        # Define keypoint pairs for drawing skeletons
        self.KEYPOINT_PAIRS = [
            (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), (6, 8),
            (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]

    def generate_frame(self):
        # Capture a frame from the camera
        while True:
            ok, frame = self.cam.read()
            if not ok:
                print('EOF')  # End of frame
                return None
            if self.src_is_cam and self.capture_queue.full():
                # drop frame
                continue
            else:
                if self.mirror:
                    frame = cv.flip(frame, 1)  # Mirror the frame if needed
                self.capture_queue.put(frame)  # Store the frame in the queue
                out, self.ratio = self.preprocess_image(frame)  # Preprocess the frame
                return out

    def preprocess_image(self, image):
        # Resize and pad the image to fit the model input shape
        h, w = image.shape[:2]
        r = min(self.model_input_shape[0] / h, self.model_input_shape[1] / w)
        image_resized = cv.resize(image, (int(w * r), int(h * r)), interpolation=cv.INTER_LINEAR)
        
        # Create a padded image
        padded_img = np.ones((self.model_input_shape[0], self.model_input_shape[1], 3), dtype=np.uint8) * 114
        padded_img[:int(h * r), :int(w * r)] = image_resized

        # Normalize image to [0, 1] range
        padded_img = padded_img / 255.0
        padded_img = padded_img.astype(np.float32)
        
        # Change the shape to (1, 3, 640, 640)
        padded_img = np.transpose(padded_img, (2, 0, 1))  # Change shape to (3, 640, 640)
        padded_img = np.expand_dims(padded_img, axis=0)  # Add batch dimension to make it (1, 3, 640, 640)
        
        return padded_img, r

    def xywh2xyxy(self, box: np.ndarray) -> np.ndarray:
        # Convert bounding boxes from [x, y, w, h] format to [x1, y1, x2, y2] format
        box_xyxy = box.copy()
        box_xyxy[..., 0] = box[..., 0] - box[..., 2] / 2
        box_xyxy[..., 1] = box[..., 1] - box[..., 3] / 2
        box_xyxy[..., 2] = box[..., 0] + box[..., 2] / 2
        box_xyxy[..., 3] = box[..., 1] + box[..., 3] / 2
        return box_xyxy

    def compute_iou(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        '''
        box and boxes are in format [x1, y1, x2, y2]
        '''
        # Calculate intersection area
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])
        inter_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

        # Calculate union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - inter_area

        return inter_area / union_area  # Return IoU

    def nms_process(self, boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
        # Apply non-maximum suppression to reduce redundant overlapping boxes
        sorted_idx = np.argsort(scores)[::-1]  # Sort scores in descending order
        keep_idx = []
        while sorted_idx.size > 0:
            idx = sorted_idx[0]  # Keep the box with the highest score
            keep_idx.append(idx)
            ious = self.compute_iou(boxes[idx, :], boxes[sorted_idx[1:], :])  # Calculate IoU
            rest_idx = np.where(ious < iou_thr)[0]  # Keep boxes with IoU below threshold
            sorted_idx = sorted_idx[rest_idx + 1]
        return keep_idx

    def process_model_output(self, *ofmaps):
        # Process model output (keypoints and bounding boxes)
        predict = ofmaps[0].squeeze(0).T  # Shape: [8400, 56]
        predict = predict[predict[:, 4] > self.box_score, :]  # Filter boxes by confidence score
        scores = predict[:, 4]
        boxes = predict[:, 0:4] / self.ratio

        boxes = self.xywh2xyxy(boxes)  # Convert bounding box format

        # Process keypoints
        kpts = predict[:, 5:]
        for i in range(kpts.shape[0]):
            for j in range(kpts.shape[1] // 3):
                if kpts[i, 3*j+2] < self.kpt_score:  # Filter keypoints by confidence score
                    kpts[i, 3*j: 3*(j+1)] = [-1, -1, -1]
                else:
                    kpts[i, 3*j] /= self.ratio
                    kpts[i, 3*j+1] /= self.ratio 
        idxes = self.nms_process(boxes, scores, self.nms_thr)  # Apply NMS
        result = {'boxes': boxes[idxes,: ].astype(int).tolist(),
                  'kpts': kpts[idxes,: ].astype(float).tolist(),
                  'scores': scores[idxes].tolist()}

        img = self.capture_queue.get()  # Get the frame from the queue
        self.capture_queue.task_done()

        # Draw keypoints and bounding boxes on the image
        color = (0,255,0)
        boxes, kpts, scores = result['boxes'], result['kpts'], result['scores']
        for  kpt, score in zip(kpts, scores):
            # Draw lines connecting keypoints
            for pair in self.KEYPOINT_PAIRS:
                pt1 = kpt[3 * pair[0]: 3 * (pair[0] + 1)]
                pt2 = kpt[3 * pair[1]: 3 * (pair[1] + 1)]
                if pt1[2] > 0 and pt2[2] > 0:
                    cv.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255, 255, 255), 3)

            # Draw individual keypoints
            for idx in range(len(kpt) // 3):
                x, y, score = kpt[3*idx: 3*(idx+1)]
                if score > 0:
                    cv.circle(img, (int(x), int(y)), 5, self.COLOR_LIST[idx % len(self.COLOR_LIST)], -1)

        self.show(img)  # Display the image
        return img

    def show(self, img):
        # Display the image in a window
        cv.imshow('Output', img)
        if cv.waitKey(1) == ord('q'):  # Exit on 'q' key press
            self.cam.release()
            cv.destroyAllWindows()
            exit(1)

def run_mxa(dfp, post_model, app):
    # Initialize the accelerator and set up model paths
    accl = AsyncAccl(dfp)
    accl.set_postprocessing_model(post_model, model_idx=0)
    accl.connect_input(app.generate_frame)
    accl.connect_output(app.process_model_output)
    accl.wait()  # Wait for the accelerator to finish

if __name__ == '__main__':
    # Parse command-line arguments for model path (-d) and post-processing ONNX file (-post)
    parser = argparse.ArgumentParser(description="Run MX3 real-time inference")
    parser.add_argument('-d', '--dfp', type=str, default="../../models/YOLO_v8_medium_pose_640_640_3_onnx.dfp", help="Specify the path to the compiled DFP file. Default is 'models/YOLO_v8_medium_pose_640_640_3_onnx.dfp'.")
    parser.add_argument('-post', '--post_model', type=str, default="../../models/YOLO_v8_medium_pose_640_640_3_onnx_post.onnx", help="Specify the path to the post model. Default is 'models/YOLO_v8_medium_pose_640_640_3_onnx_post.onnx.")
    args = parser.parse_args()

    # Connect to the camera and initialize the app
    cam = cv.VideoCapture(0)
    parent_path = Path(__file__).resolve().parent
    model_input_shape = (640, 640)
    app = App(cam, model_input_shape, mirror=False, src_is_cam=True)
    dfp = args.dfp
    post_model = args.post_model
    run_mxa(dfp, post_model, app)
