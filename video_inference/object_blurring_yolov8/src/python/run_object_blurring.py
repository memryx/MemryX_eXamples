# OpenCV and helper libraries imports
import sys
import os
from pathlib import Path
from queue import Queue
import cv2 as cv
import numpy as np
from memryx import AsyncAccl
from typing import List
import argparse


class App:
    def __init__(self, cam, model_input_shape, output_path, mirror=False, src_is_cam=False, **kwargs):
        # Initialize camera and various configurations
        self.cam = cam
        self.input_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.input_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
        self.model_input_shape = model_input_shape
        self.capture_queue = Queue(maxsize=10)  # Queue to store frames for processing
        self.mirror = mirror  # Flag to mirror the video frame
        self.confidence_thres = 0.25  # Threshold for object confidence
        self.iou_thres = 0.7  # IoU threshold for non-max suppression
        self.src_is_cam = src_is_cam

        # Calculate the scaling factors for the bounding box coordinates
        input_max_dim = max((self.input_height, self.input_width))
        self.x_factor = input_max_dim / self.model_input_shape[1]
        self.y_factor = input_max_dim / self.model_input_shape[0]

        ### Constants
        self.bbox_thickness = 4
        self.bbox_color = (0, 255, 0)  # green color

        # .............................
        # Initialize video writer
        fc = cv.VideoWriter_fourcc(*'mp4v')  # Codec
        self.vw = cv.VideoWriter(output_path, fc, cam.get(cv.CAP_PROP_FPS),
            (self.input_width, self.input_height))
        # .............................

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
        padded_img = np.zeros((self.model_input_shape[0], self.model_input_shape[1], 3), dtype=np.uint8)
        padded_img[:int(h * r), :int(w * r)] = image_resized

        # Normalize image to [0, 1] range
        #
        # use_model_shape[input]=False means we can keep the image as HWC,
        # since both opencv and native memryx use HWC order
        padded_img = padded_img.astype(np.float32) / 255.0
        
        return padded_img, r

    def xywh2xyxy(self, box: np.ndarray) -> np.ndarray:
        # Convert bounding boxes from [x, y, w, h] format to [x1, y1, x2, y2] format
        box_xyxy = box.copy()
        box_xyxy[..., 0] = (box[..., 0] - box[..., 2] / 2) * self.x_factor
        box_xyxy[..., 1] = (box[..., 1] - box[..., 3] / 2) * self.y_factor
        box_xyxy[..., 2] = (box[..., 0] + box[..., 2] / 2) * self.x_factor
        box_xyxy[..., 3] = (box[..., 1] + box[..., 3] / 2) * self.y_factor
        return box_xyxy

    def postprocess(self, output):
        # Transpose the output to shape (8400, 84)
        outputs = np.transpose(np.squeeze(output[0]))

        # Extract the bounding box information and class scores in a vectorized manner
        boxes = outputs[:, :4]  # (8400, 4) - x_center, y_center, width, height
        person_scores = outputs[:, 4]  # Only look at person class

        # Filter out detections with scores below the confidence threshold
        valid_indices = np.where(person_scores >= self.confidence_thres)[0]
        if len(valid_indices) == 0:
            return []  # Return an empty list if no valid detections

        # Select only valid detections
        valid_boxes = boxes[valid_indices]
        valid_scores = person_scores[valid_indices]

        # Convert bounding box coordinates from (x_center, y_center, w, h) to (x1, y1, x2, y2)
        valid_boxes = self.xywh2xyxy(valid_boxes)

        # Create detection dictionaries
        detections = [{
            'bbox': valid_boxes[i].astype(int).tolist(),
            'class_id': 0, # Person class id is 0
            'class': 'person',
            'score': valid_scores[i]
        } for i in range(len(valid_indices))]

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        if len(detections) > 0:
            # NMS requires two lists: bounding boxes and confidence scores
            boxes_for_nms = [d['bbox'] for d in detections]
            scores_for_nms = [d['score'] for d in detections]

            # Use OpenCV's NMS function, which is pretty efficient
            indices = cv.dnn.NMSBoxes(boxes_for_nms, scores_for_nms, self.confidence_thres, self.iou_thres)

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

    def blur_roi(self, frame, x1, y1, x2, y2, ksize=35):
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return frame

        blurred = cv.GaussianBlur(roi, (ksize | 1, ksize | 1), 0)
        frame[y1:y2, x1:x2] = blurred
        return frame

    def process_model_output(self, *ofmaps):
        results = self.postprocess(ofmaps)  # Postprocess the model output

        img = self.capture_queue.get()  # Get the frame from the queue
        self.capture_queue.task_done()

        for det in results:
            x1, y1, x2, y2 = det['bbox']
            blurred_img = self.blur_roi(img, x1, y1, x2, y2, ksize=75)
            cv.rectangle(
                blurred_img,
                (x1, y1),
                (x2, y2),
                self.bbox_color,
                thickness=self.bbox_thickness
            )

        self.show(blurred_img)  # Optional: display processed frame.
        self.vw.write(blurred_img)  # Write the results

        return img

    def show(self, img):
        # Display the image in a window
        cv.imshow('Output', img)
        if cv.waitKey(1) == ord('q'):  # Exit on 'q' key press
            self.cam.release()
            cv.destroyAllWindows()
            self.vw.release()
            exit(1)

def run_mxa(dfp, post_model, app):
    # Initialize the accelerator and set up model paths
    #
    #
    # We use use_model_shape=(input=False,output=True) to indicate that
    # we will force NHWC input order, even though the ONNX model is NCHW.
    #
    # This is an optimization to lessen the number of tranposes needed
    # when working with OpenCV inputs (which are always HWC).
    #
    # Output is left as the original model shape
    accl = AsyncAccl(dfp, use_model_shape=(False,True))
    accl.set_postprocessing_model(post_model, model_idx=0)
    accl.connect_input(app.generate_frame)
    accl.connect_output(app.process_model_output)
    accl.wait()  # Wait for the accelerator to finish

if __name__ == '__main__':
    # Parse command-line arguments for model path (-d) and post-processing ONNX file (-post)
    parser = argparse.ArgumentParser(description="Run MX3 real-time inference")
    parser.add_argument('-d', '--dfp', type=str, default="../models/YOLO_v8_small_640_640_3_onnx.dfp", help="Specify the path to the compiled DFP file. Default is 'models/YOLO_v8_small_640_640_3_onnx.dfp'.")
    parser.add_argument('-post', '--post_model', type=str, default="../models/YOLO_v8_small_640_640_3_onnx_post.onnx", help="Specify the path to the post model. Default is 'models/YOLO_v8_small_640_640_3_onnx_post.onnx.")
    args = parser.parse_args()

    # Connect to the camera and initialize the app
    video_path = "../videos/sample-1.mov"
    cam = cv.VideoCapture(video_path)
    parent_path = Path(__file__).resolve().parent
    model_input_shape = (640, 640)

    output_path = f"results.mp4"

    app = App(cam, model_input_shape, output_path=output_path, 
                mirror=False, src_is_cam=False)
    dfp = args.dfp
    post_model = args.post_model
    run_mxa(dfp, post_model, app)
