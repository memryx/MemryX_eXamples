from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import cv2 as cv
import numpy as np
from datetime import datetime
import multiprocessing as mp
import time
from pose import Pose, Part, PoseSequence
from memryx import AsyncAccl
import sys
import os
from pathlib import Path
from queue import Queue, Empty
import torchvision.ops as ops
from typing import List

from multiprocessing import Value
from tracker.byte_tracker import BYTETracker
from argparse import Namespace
import torch

# Initialize FastAPI and Jinja2Templates
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="./templates")

# Global variables
globexercise = 'bicepcurl'
vid_rec_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
frame_queue = mp.Queue()  # Multiprocessing Queue for frame sharing
inference_process = None
poseapp = None
run_flag = Value('b', False) 

class poseApp:
    COLOR_LIST = list([[128, 255, 0], [255, 128, 50], [128, 0, 255], [255, 255, 0],
                   [255, 102, 255], [255, 51, 255], [51, 153, 255], [255, 153, 153],
                   [255, 51, 51], [153, 255, 153], [51, 255, 51], [0, 255, 0],
                   [255, 0, 51], [153, 0, 153], [51, 0, 51], [0, 0, 0],
                   [0, 102, 255], [0, 51, 255], [0, 153, 255], [0, 153, 153]])

        # Define keypoint pairs for drawing skeletons
    KEYPOINT_PAIRS = [
            (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), (6, 8),
            (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]
    
    EXERCISE_CFG = {
                    "bicep_curl" : {"up_angle": 60, "down_angle":120, "l_parts": ['lshoulder','lelbow', 'lwrist'], "r_parts": ['rshoulder','relbow', 'rwrist'], "text": "Make sure atleast one arm and your hip is visible"},
                    "front_raises" : {"up_angle": 60, "down_angle":120, "l_parts": ['lwrist', 'lshoulder', 'lhip', 'lelbow'], "r_parts": ['rwrist', 'rshoulder', 'rhip', 'relbow'], "text": "Make sure atleast one arm and your hip is visible"},
                    "shoulder_press" : {"up_angle": 140, "down_angle":60, "l_parts": ['lwrist', 'lelbow', 'lshoulder'], "r_parts": ['rwrist', 'relbow', 'rshoulder'], "text": "Make sure atleast one arm and your hip is visible"},
                    "squats" : {"up_angle": 140, "down_angle":60, "l_parts": ['lhip', 'lknee', 'lankle'], "r_parts": ['rhip', 'rknee', 'rankle'], "text": "Make sure atleast one arm and your hip is visible"}
                }

    BYTETRACK_CFG = {
                    "track_thresh" : 0.5, # High_threshold
                    "track_buffer" : 50, # Number of frame lost tracklets are kept
                    "match_thresh": 0.9,    # Matching threshold for bounding boxes
                    "track_buffer": 30,     # Number of frames to keep track without new detection
                    "mot20": True          # Use settings for MOT20 dataset (if using MOT20)
                    }

    

    def __init__(self, dfp, post_model, model_input_shape, frame_queue,run_flag, exercise, mirror=False):
        # self.cam = cam
        self.cam = cv.VideoCapture(0)
        if not self.cam.isOpened():
            print("Unable to read camera feed")
            return
        self.cam.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
        self.cam.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
        self.input_height = int(self.cam.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.input_width = int(self.cam.get(cv.CAP_PROP_FRAME_WIDTH))
        print("CAM Height and Width ", self.input_height, self.input_width)
        self.model_input_shape = model_input_shape
        self.capture_queue = Queue(maxsize=5)
        # self.pose_output_queue = deque(maxlen=2)
        self.mirror = mirror
        self.box_score = 0.25
        self.kpt_score = 0.5
        self.nms_thr = 0.2
        self.running = run_flag
        self.ratio = None
        self.output_frame_queue = frame_queue

        self.accl = None
        self.dfp = dfp
        self.post_model = post_model
        self.exercise = exercise
        self.counts = []
        self.stages = []
        self.angles = []
        self.last_stage_change = []

        tracker_config = Namespace(**self.BYTETRACK_CFG)
        self.tracker = BYTETracker( tracker_config, frame_rate=30)
        self.tracked_objects = None
        self.searchtext = ""
        self.searchtxtpos = (int(self.input_width/2), int(self.input_height/2))
        
        def cal_image_center():
            text = "Searching for people"
            font = cv.FONT_HERSHEY_SIMPLEX
            scale = 2
            thickness = 2

            # Get the text size
            (text_width, text_height), baseline = cv.getTextSize(text, font, scale, thickness)

            # Calculate the center position
            x = (self.input_width - text_width) // 2
            y = (self.input_height + text_height) // 2
            self.searchtxtpos = (x,y)
            
        
        cal_image_center()

    def generate_frame(self):
        while True:
            ok, frame = self.cam.read()
            # print(frame.shape)
            if not self.running.value:
                print('EOF')
                # self.cam.release() 
                self.stop()
                return None
            else:
                if self.capture_queue.full():
                    # drop frame
                    continue
                else:
                    if self.mirror:
                        frame = cv.flip(frame, 1)
                    self.capture_queue.put(frame)
                    out, self.ratio = self.preprocess_image(frame)
                    return out

    def preprocess_image(self, image):
        h, w = image.shape[:2]
        r = min(self.model_input_shape[0] / h, self.model_input_shape[1] / w)
        image_resized = cv.resize(image, (int(w * r), int(h * r)), interpolation=cv.INTER_LINEAR)

        # Create padded image
        padded_img = np.full((self.model_input_shape[0], self.model_input_shape[1], 3), 114, dtype=np.uint8)
        padded_img[:int(h * r), :int(w * r)] = image_resized

        # Normalize to [0, 1]
        padded_img = (padded_img / 255.0).astype(np.float32)
        
        # Add batch dimension
        padded_img = np.expand_dims(padded_img, axis=2)
        return padded_img, r

    def xywh2xyxy(self, box: np.ndarray) -> np.ndarray:
        box_xyxy = np.copy(box)
        box_xyxy[:, 0] = box[:, 0] - box[:, 2] / 2
        box_xyxy[:, 1] = box[:, 1] - box[:, 3] / 2
        box_xyxy[:, 2] = box[:, 0] + box[:, 2] / 2
        box_xyxy[:, 3] = box[:, 1] + box[:, 3] / 2
        return box_xyxy

    def compute_iou(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])

        inter_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - inter_area

        return inter_area / union_area

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
        
                
        try:
            img = self.capture_queue.get()
        except Empty:
            return None
        self.capture_queue.task_done()

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

        # Draw keypoints and bounding boxes on the image
        # color = (0,255,0)
        boxes, kpts, scores = result['boxes'], result['kpts'], result['scores']
        # print("scores:::::", scores)

        byte_detections = []
        for  kpt, score, bbox in zip(kpts, scores, boxes):
            x_min, y_min, x_max, y_max = bbox
            byte_detections.append([x_min, y_min, x_max, y_max, score, 0.7, 0.7, 0])

        # Apply tracker

        try:

            self.tracked_objects = self.tracker.update(torch.tensor(byte_detections))

            # Adding new human

            if len(self.tracked_objects) > 0:
                if len(self.tracked_objects) > len(self.counts):
                    new_human = len(self.tracked_objects) - len(self.counts)
                    self.angles += [0] * new_human
                    self.counts += [0] * new_human
                    self.stages += ["-"] * new_human
                    self.last_stage_change += [time.time()] * new_human
                # if len(self.tracked_objects) == 0:
                #     track_id = 0

                # enumerateover track
                for index, track in enumerate(self.tracked_objects):
                    track_id = track.track_id
                    kpt = kpts[index]
                    box = boxes[index]

                    text_position = (box[2] - box[0], box[3] - box[1])

                    for pair in self.KEYPOINT_PAIRS:
                        pt1 = kpt[3 * pair[0]: 3 * (pair[0] + 1)]
                        pt2 = kpt[3 * pair[1]: 3 * (pair[1] + 1)]

                        if pt1[2] > 0 and pt2[2] > 0:
                            cv.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255, 255, 255), 3)

                    # Draw individual keypoints

                    parts_kpt = []
                    for idx in range(len(kpt) // 3):
                        # print("id :: ", idx , " ::: ", kpt[3*idx: 3*(idx+1)])
                        x, y, score = kpt[3*idx: 3*(idx+1)]
                        parts_kpt.append([x,y,score])
                        if score > 0:
                            cv.circle(img, (int(x), int(y)), 5, self.COLOR_LIST[idx % len(self.COLOR_LIST)], -1)

                    pose_cl = Pose(parts_kpt)
                    # pose_str = Pose
                    # print(pose_cl.print(Pose.PART_NAMES))
                    right, left = self.is_necessary_kpt_visible(pose_cl)
                    # print("side visible :: Right :: ",right , " Left :: ", left)
                    if not right and not left:
                        cv.putText(img = img,text=f'ID: {track_id}', org = (text_position[0], text_position[1] - 20), 
                            fontFace = cv.FONT_HERSHEY_DUPLEX,
                            fontScale = 0.6,
                            color = (0, 255, 0),
                            thickness = 2
                            )
                        cv.putText(img = img,text="Keypoints not visible", org = text_position, 
                                fontFace = cv.FONT_HERSHEY_DUPLEX,
                                fontScale = 0.6,
                                color = (0, 0, 255),
                                thickness = 2
                                )
                    else:
                        if right:
                            a = pose_cl.get_part_xy(self.EXERCISE_CFG[self.exercise]['r_parts'][0])
                            b = pose_cl.get_part_xy(self.EXERCISE_CFG[self.exercise]['r_parts'][1])
                            c = pose_cl.get_part_xy(self.EXERCISE_CFG[self.exercise]['r_parts'][2])

                            self.angles[index] = self.estimate_pose_angle(a,b,c)
                            text_position = (int(b[0]), int(b[1]))
                        else:
                            a = pose_cl.get_part_xy(self.EXERCISE_CFG[self.exercise]['l_parts'][0])
                            b = pose_cl.get_part_xy(self.EXERCISE_CFG[self.exercise]['l_parts'][1])
                            c = pose_cl.get_part_xy(self.EXERCISE_CFG[self.exercise]['l_parts'][2])

                            self.angles[index] = self.estimate_pose_angle(a,b,c) 
                            text_position =  (int(b[0]), int(b[1]))       


                        
                        self.repcounter(index)

                        count_txt = "Rep Count = "+ str(self.counts[index])
                        cv.putText(img = img,text=f'ID: {track_id}', org = (text_position[0], text_position[1] - 20), 
                            fontFace = cv.FONT_HERSHEY_DUPLEX,
                            fontScale = 0.6,
                            color = (0, 255, 0),
                            thickness = 2
                            )
                        cv.putText(img = img,text = count_txt, org = text_position, 
                                    fontFace = cv.FONT_HERSHEY_DUPLEX,
                                    fontScale = 0.6,
                                    color = (0, 255, 0),
                                    thickness = 2
                                    )
            else:

                
                self.searchtext = "" if self.searchtext == "Searching for People" else "Searching for People"
                cv.putText(img = img,text = self.searchtext, org = self.searchtxtpos, 
                                    fontFace = cv.FONT_HERSHEY_DUPLEX,
                                    fontScale = 2,
                                    color = (0, 0, 255),
                                    thickness = 2
                                    )

        except Exception as e:
            print(f"An error occurred while updating tracker: {e}")
            self.tracked_objects = None  # Or set to a default value if necessary
            self.running.value = False
        

        # self.show(img)
        ret, buffer = cv.imencode('.jpg', img)
        self.output_frame_queue.put(buffer)
        
        return img
    
    def repcounter(self, index):
        current_time = time.time()
        
        # Check if stage remains the same for more than 3 seconds
        if current_time - self.last_stage_change[index] > 3:
            self.counts[index] = 0  # Reset count if no change in stage for more than 3 seconds

        # If angle is less than up_angle, update to "up" and increment count if coming from "down"
        if self.angles[index] < self.EXERCISE_CFG[self.exercise]["up_angle"]:
            if self.stages[index] == "down":
                self.counts[index] += 1
            if self.stages[index] != "up":
                self.stages[index] = "up"
                self.last_stage_change[index] = current_time  # Reset last stage change time

        # If angle is more than down_angle, update to "down"
        elif self.angles[index] > self.EXERCISE_CFG[self.exercise]["down_angle"]:
            if self.stages[index] != "down":
                self.stages[index] = "down"
                self.last_stage_change[index] = current_time  # Reset last stage change time



    def estimate_pose_angle(self, a, b, c):
        """
        Calculate the pose angle for object.

        Args:
            a (float) : The value of pose point a
            b (float): The value of pose point b
            c (float): The value o pose point c

        Returns:
            angle (degree): Degree value of angle between three points
        """
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def is_necessary_kpt_visible(self, pose):

        right_present = pose.if_given_parts_exists(self.EXERCISE_CFG[self.exercise]['r_parts'])
        left_present = pose.if_given_parts_exists(self.EXERCISE_CFG[self.exercise]['l_parts'])

        return right_present, left_present
        


    def stop(self):
        print("Stopping the poseApp")
        self.cam.release()  # Release camera resources

        while(not self.output_frame_queue.empty()):
            print("flushing queue in child process ")
            _= self.output_frame_queue.get()

    def start(self):
        if not self.running.value or not self.cam.isOpened():
            self.cam = cv.VideoCapture(0)
            self.running.value = True
            print("Reopened camera")

        self.accl = AsyncAccl(self.dfp)
        self.accl.set_postprocessing_model(self.post_model, model_idx=0)
        self.accl.connect_input(self.generate_frame)
        self.accl.connect_output(self.process_model_output)
        self.accl.wait()

    def show(self, img):
        # Display the image in a window
        cv.imshow('Output', img)
        if cv.waitKey(1) == ord('q'):  # Exit on 'q' key press
            cv.destroyAllWindows()
            self.cam.release()
            exit(1)


def run_mxa(_run_flag, p_poseapp):
    global run_flag

    # Continuously process frames and add them to the queue
    p_poseapp.start()
    print("Pose Application started " )

    # Sleep for safety when proces ends
    # while run_flag.value:
    #     time.sleep(1)


def display_pose():
    global frame_queue, run_flag

    while run_flag.value:
        try:
            buffer = frame_queue.get(timeout=1)  # Add timeout to avoid blocking indefinitely
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Empty:
            # Skip frame if the queue is empty to avoid blocking
            continue

def start_inference(exercise):
    global inference_process, frame_queue, poseapp, run_flag
    run_flag.value = True
    poseapp = poseApp("../models/YOLO_v8_medium_pose_640_640_3_onnx.dfp", "../models/YOLO_v8_medium_pose_640_640_3_onnx_post.onnx", (640, 640), frame_queue, run_flag, exercise, mirror=False)
    if inference_process is None or not inference_process.is_alive():
        inference_process = mp.Process(
            target=run_mxa,
            args=(run_flag, poseapp),
            daemon=True
        )
        inference_process.start()

def stop_inference():
    global inference_process, poseapp, run_flag, frame_queue
    run_flag.value = False
    if not frame_queue.empty():
        while(not frame_queue.empty()):
            print("flushing in main process ")
            _ = frame_queue.get()
    print("Stop inference is called \n")

    if poseapp is not None:
        print("pose app stop is called\n")
        time.sleep(1)  # Short delay to ensure resources are release

    if inference_process is not None:
        print("Joining the process ")
        inference_process.join()  # Wait a short time for it to join
        inference_process = None

@app.get("/", response_class=HTMLResponse)
@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    stop_inference()
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/bicepcurl", response_class=HTMLResponse)
async def bicepcurl(request: Request):
    if(run_flag.value):
        stop_inference()
    else:
        start_inference("bicep_curl")
    return templates.TemplateResponse("bicepcurl.html", {"request": request})

@app.get("/frontraise", response_class=HTMLResponse)
async def frontraise(request: Request):
    if(run_flag.value):
        stop_inference()
    else:
        start_inference("front_raises")
    return templates.TemplateResponse("frontraise.html", {"request": request})

@app.get("/squats", response_class=HTMLResponse)
async def shouldershrug(request: Request):
    if(run_flag.value):
        stop_inference()
    else:
        start_inference("squats")
    return templates.TemplateResponse("squats.html", {"request": request})

@app.get("/shoulderpress", response_class=HTMLResponse)
async def shoulderpress(request: Request):
    if(run_flag.value):
        stop_inference()
    else:
        start_inference("shoulder_press")
    return templates.TemplateResponse("shoulderpress.html", {"request": request})



@app.get('/video_rec/{exercise}')
async def video_rec(exercise: str):
    return StreamingResponse(display_pose(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get('/stop_inference')
async def stop_inference_route():
    stop_inference()
    return JSONResponse({"status": "stopped"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
