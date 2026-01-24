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
import supervision as sv
from collections import defaultdict
from ultralytics.utils.plotting import colors


COCO_CLASSES = ("person", "bicycle", "car", "motorcycle", "airplane", "bus",
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
        "scissors", "teddy bear", "hair drier", "toothbrush")

class App:
    def __init__(self, cam, model_input_shape, output_path, mirror=False, src_is_cam=False, save_output=False, show_output=True, **kwargs):
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
        self.save_output = save_output
        self.show_output = show_output

        # Calculate the scaling factors for the bounding box coordinates
        input_max_dim = max((self.input_height, self.input_width))
        self.x_factor = input_max_dim / self.model_input_shape[1]
        self.y_factor = input_max_dim / self.model_input_shape[0]

        # Init bytetrack
        self.tracker = sv.ByteTrack()
        self.track_history = defaultdict(list)
        self.track_last_side = {}   # tid -> -1 or +1

        ### Constants
        self.bbox_thickness = 6
        self.text_scale = 1.6
        self.text_thickness = 2
        self.text_offset_y = 12
        self.text_padding = 20
        self.circle_radius = 14
        self.polyline_thickness = 3
        self.text_color = (255, 255, 255)  # White
        self.line_color = (104, 31, 17) # navyblue
        self.line_thickness = 9

        # Line Counter
        self.line_p1 = (1880, 40)   # <-- adjustable
        self.line_p2 = (40, 1040)   # <-- adjustable
        self.count_in = 0
        self.count_out = 0
        self.counted_ids = set()   # avoid double counting
        

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

    def crossed_line(self, prev_pt, curr_pt, eps=1e-6):
        s1 = self.side_of_line(prev_pt, self.line_p1, self.line_p2)
        s2 = self.side_of_line(curr_pt, self.line_p1, self.line_p2)
        if abs(s1) < eps or abs(s2) < eps:
            return False  # touching line ≠ crossing
        return s1 * s2 < 0

    def side_of_line(self, p, a, b):
        """
        Returns:
            >0 if point p is on one side
            <0 if on the other side
            =0 if exactly on the line
        """
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

    def preprocess_image(self, image):
        # Resize and pad the image to fit the model input shape
        h, w = image.shape[:2]
        r = min(self.model_input_shape[0] / h, self.model_input_shape[1] / w)
        image_resized = cv.resize(image, (int(w * r), int(h * r)), interpolation=cv.INTER_LINEAR)
        
        # Create a padded image
        padded_img = np.zeros((self.model_input_shape[0], self.model_input_shape[1], 3), dtype=np.uint8)
        padded_img[:int(h * r), :int(w * r)] = image_resized

        # OpenCV uses BGR channels by default, but the model expects RGB
        padded_img = cv.cvtColor(padded_img, cv.COLOR_BGR2RGB)

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

        valid_boxes = self.xywh2xyxy(valid_boxes)

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

    def to_sv_detections(self, detections):
        if len(detections) == 0:
            return sv.Detections.empty()

        xyxy = np.array([d["bbox"] for d in detections], dtype=np.float32)
        confidence = np.array([d["score"] for d in detections], dtype=np.float32)
        class_id = np.array([d["class_id"] for d in detections], dtype=np.int64)

        return sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )

    def draw_counter_box(self, img, text, top_right, bg_color=(0, 0, 0), text_color=(255, 255, 255)):
        font, font_scale, thickness, padding =cv.FONT_HERSHEY_SIMPLEX, 2, 5, 20
        (tw, th), baseline = cv.getTextSize(text, font, font_scale, thickness)
        x_right, y = top_right
        box_w = tw + padding * 2
        box_h = th + baseline + padding * 2
        box_x1 = x_right - box_w
        box_y1 = y
        box_x2 = x_right
        box_y2 = y + box_h

        # Draw background box
        cv.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), bg_color, -1)
        center_x, center_y = (box_x1 + box_x2) // 2, (box_y1 + box_y2) // 2
        text_x, text_y = center_x - tw // 2, center_y + th // 2
        cv.putText(img, text, (text_x, text_y), font, font_scale, text_color,
                   thickness, cv.LINE_AA)


    def process_model_output(self, *ofmaps):
        results = self.postprocess(ofmaps)  # Postprocess the model output

        img = self.capture_queue.get()  # Get the frame from the queue
        self.capture_queue.task_done()
        
        tracks = self.tracker.update_with_detections(self.to_sv_detections(results))

        for xyxy, cls, tid in zip(tracks.xyxy, tracks.class_id, tracks.tracker_id):
            if int(cls) in {24, 26, 28}: # luggage class in COCO dataset
                x1, y1, x2, y2 = map(int, xyxy)

                # color for each class. 
                box_color = colors(16, bgr=True)

                # Store tracks
                x, y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                track = self.track_history[tid]
                track.append((float(x), float(y)))
                if len(track) > 45:  # store last 30 frames track position
                    track.pop(0)
                
                # ---- Object counting ----
                curr_pt = (x, y)
                side = self.side_of_line(curr_pt, self.line_p1, self.line_p2)
                DEAD_ZONE = 15
                if abs(side) < DEAD_ZONE:  # Ignore points too close to the line
                    continue
                side = 1 if side > 0 else -1
                # First time seeing this track
                if tid not in self.track_last_side:
                    self.track_last_side[tid] = side
                    continue
                # Side changed → crossed
                if side != self.track_last_side[tid]:
                    # Direction (normal-based, line-agnostic)
                    dx = self.line_p2[0] - self.line_p1[0]
                    dy = self.line_p2[1] - self.line_p1[1]
                    nx, ny = -dy, dx  # normal
                    mvx, mvy = curr_pt[0] - track[-2][0], curr_pt[1] - track[-2][1]
                    direction = mvx * nx + mvy * ny
                    if direction > 0:
                        self.count_in += 1
                    else:
                        self.count_out += 1
                    self.counted_ids.add(tid)  # Append counted
                self.track_last_side[tid] = side  # Update last side

                # Draw bounding box
                cv.rectangle(img,(x1, y1),(x2, y2),box_color, thickness=self.bbox_thickness)

                # Draw bounding box label
                label = f"{COCO_CLASSES[int(cls)]}"
                (text_w, text_h), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX,
                                                self.text_scale, self.text_thickness)
                cv.rectangle(img, (x1, y1 - text_h - self.text_padding * 2),
                            (x1 + text_w + self.text_padding * 2, y1), box_color, -1)            
                cv.putText(img,label, (x1 + self.text_padding, y1 - self.text_padding),
                            cv.FONT_HERSHEY_SIMPLEX, self.text_scale,
                            self.text_color, self.text_thickness, cv.LINE_AA)
        
        # Display counting line and results
        cv.line(img, self.line_p1, self.line_p2, self.line_color, self.line_thickness)
        cv.circle(img, self.line_p1, self.circle_radius, self.line_color, -1)
        cv.circle(img, self.line_p2, self.circle_radius, self.line_color, -1)
        h, w = img.shape[:2]
        margin = 20
        self.draw_counter_box(img, f"IN : {self.count_in}",
                                    top_right=(w - margin, margin),
                                    bg_color=(108, 27, 255))
        self.draw_counter_box(img, f"OUT: {self.count_out}",
                                    top_right=(w - margin, margin + 120),
                                    bg_color=(104, 31, 17))

        if self.show_output:
            self.show(img)  # display processed frame.

        if self.save_output and self.vw is not None:
            self.vw.write(img)  # Write the results

        return img

    def show(self, img):
        # Display the image in a window
        cv.namedWindow('Output', cv.WINDOW_NORMAL)
        cv.imshow('Output', img)
        if cv.waitKey(1) == ord('q'):  # Exit on 'q' key press
            self.cam.release()
            cv.destroyAllWindows()
            if self.save_output and self.vw is not None:
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
    parser = argparse.ArgumentParser(description="Run MX3 real-time luggage counting using YOLOv8.")
    parser.add_argument('-d', '--dfp', type=str, default="../models/YOLO_v8_small_640_640_3_onnx.dfp", help="Specify the path to the compiled DFP file. Default is '../models/YOLO_v8_small_640_640_3_onnx.dfp'.")
    parser.add_argument('-post', '--post_model', type=str, default="../models/YOLO_v8_small_640_640_3_onnx_post.onnx", help="Specify the path to the post model. Default is '../models/YOLO_v8_small_640_640_3_onnx_post.onnx.")
    parser.add_argument('-s', '--save', action='store_true', help="Enable saving output to file. Output will be ./results.mp4  Default is False.")
    parser.add_argument('-m', '--mirror', action='store_true', help="Mirror the video horizontally. Useful for webcam input.")
    parser.add_argument('-c', '--cam', action='store_true', help="Use the camera as input source (will use opencv camera #0).")
    parser.add_argument('-v', '--video', type=str, default="", help="Use a video file as input source, or a camera full path for non-index-0 cams (e.g., /dev/video2).")
    parser.add_argument('--no_show', action='store_false', help="Disable displaying output window. Useful when working with video files.")

    args = parser.parse_args()

    # User needs to specify either camera or video file
    if args.cam:
        cam = cv.VideoCapture(0)
        if not cam.isOpened():
            print("Error: Could not open camera 0")
            sys.exit(1)
    elif args.video != "":
        cam = cv.VideoCapture(args.video)
        if not cam.isOpened():
            print(f"Error: Could not open video file {args.video}")
            sys.exit(1)
    else:
        print("Error: Please specify either --cam to use the camera or --video <path> to use a video file.")
        cam = None
        sys.exit(1)

    # Constants
    model_input_shape = (640, 640)
    output_path = "results.mp4"

    # Connect to the camera and initialize the app
    app = App(cam, model_input_shape, output_path=output_path, 
                mirror=args.mirror, src_is_cam=args.cam,
                save_output=args.save, show_output=args.no_show)
    dfp = args.dfp
    post_model = args.post_model
    run_mxa(dfp, post_model, app)