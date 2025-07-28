"""
============
Information:
============
Project: Intrusion Detection using ByteTrack and YOLOv8m example code on MXA
File Name:  intrusion_demo.py
"""

###################################################################################################

# Imports
import argparse
import numpy as np
import cv2
from queue import Queue,Full
from memryx import AsyncAccl
from yolov8.yolov8 import YoloV8 as YoloModel
from yolov8.tracker.byte_tracker import BYTETracker
from yolov8.timer import Timer
import torch
from yolov8.visualize import plot_tracking

DEFAULT_ROI = [1000, 500, 1850,800]
ALERT_NUM_FRAMES = 60
INIT_NUM_FRAMES = 20

###################################################################################################
###################################################################################################
###################################################################################################

def make_parser():
    parser = argparse.ArgumentParser("MX Intrusion Demo!")
    parser.add_argument('--roi_coordinates', nargs='+', type=int, help='space seperated ROI coordinates in x1y1x2y2 format')
    parser.add_argument("--dfp", default="../../models/yolov8m_float32.dfp", type=str, help="dfp path")
    parser.add_argument("--post_model_path",default="../../models/yolov8m_float32_post.tflite", type=str, help="postprocessing model path")
    parser.add_argument(
        "--input_path", required=True, help="path to video"
    )
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser

def inRoi(xywh,roi):
    """
    Check where the given rectangle is in the ROI
    """
    x1,y1,w,h = xywh
    x2 = x1+w
    y2 = y1+h
    centroid_box = [(x1+x2)/2,(y1+y2)/2]
    if centroid_box[0]>roi[0] and centroid_box[0]<roi[2] and centroid_box[1]>roi[1] and centroid_box[1]<roi[3]:
        return True
    return False

class IntrusionMxa:
    """
    A demo app to run YOLOv8m on the MemryX MXA.
    """

###################################################################################################
    def __init__(self,args,show):
        """
        The initialization function.
        """

        # Controls
        self.args = args
        self.show = show
        self.dfp = args.dfp
        self.post_model = args.post_model_path
        self.cap_queue = Queue(maxsize=5)
        if "/dev/video" in str(args.input_path):
            self.src_is_cam = True
        else:
            self.src_is_cam = False
        self.vcap = cv2.VideoCapture(args.input_path)
        self.tracker = BYTETracker(args, frame_rate=args.fps)
        self.timer = Timer()
        self.frame_id = 0
        self.dims = (int(self.vcap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(self.vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # Initialize the model with the stream dimensions
        self.model = YoloModel(stream_img_size=(self.dims[1], self.dims[0], 3))
        self.done = False
        self.color_wheel = np.random.randint(0, 255, (20, 3)).astype(np.int32)
        self.object_store = {}
        if args.roi_coordinates and len(args.roi_coordinates)==4:
            self.roi = args.roi_coordinates
        else:
            self.roi = DEFAULT_ROI
        self.detect_counter = 0
        self.roi_color = (0,255,0)

###################################################################################################
    def run(self):
        """
        The function that starts the inference on the MXA.
        """
        accl = AsyncAccl(dfp=self.dfp)
        accl.set_postprocessing_model(self.post_model, model_idx=0)

        # Connect the input and output functions and let the accl run
        accl.connect_input(self.capture_and_preprocess)
        accl.connect_output(self.postprocess)
        accl.wait()

        self.done = True

        cv2.destroyAllWindows()
        self.vcap.release()

###################################################################################################
    def capture_and_preprocess(self):
        """
        Captures a frame for the video device and pre-processes it.
        """
        while True:
            got_frame, frame = self.vcap.read()

            if not got_frame or self.done:
                return None

            if self.src_is_cam and self.cap_queue.full():
                # drop cam frame
                continue
            else:
                # Put the frame in the cap_queue to be processed later
                self.cap_queue.put(frame)

                # Pre-process the frame using the corresponding model
                frame = self.model.preprocess(frame)
                return frame

###################################################################################################
    def postprocess(self, *mxa_output):
        """
        Post-process the MXA output.
        """
        dets = self.model.postprocess(mxa_output)
        # Push the detection results to the queue
        frame = self.cap_queue.get()
        self.cap_queue.task_done()
        if not dets:
            return

        # Draw detection boxes on the frame
        detection_results = []
        for d in dets:

            x1, y1, w, h = d['bbox']
            x2,y2 = x1+w,y1+h
            result =[x1,y1,x2,y2,d['score']]
            detection_results.append(result)
        detection_results = torch.tensor(detection_results)
        online_targets = self.tracker.update(detection_results)
        online_tlwhs = []
        online_ids = []
        online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > self.args.aspect_ratio_thresh
            if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
        if self.timer.start_time>0:
            self.timer.toc()
        else:
            self.timer.average_time = 10000
        if self.frame_id<INIT_NUM_FRAMES:
            for i,id in enumerate(online_ids):
                if inRoi(online_tlwhs[i],self.roi):
                    self.object_store[id] = online_tlwhs[i]
        else:
            for i,id in enumerate(online_ids):
                if id not in self.object_store and inRoi(online_tlwhs[i],self.roi):
                    print("new object with id: ",id)
                    self.roi_color = (0, 0, 255)
                    self.detect_counter = 1
                    self.object_store[id] = online_tlwhs[i]
        overlay = frame.copy()
        alpha = 0.5 
        if self.detect_counter==0 or self.detect_counter>ALERT_NUM_FRAMES:
            self.roi_color = (0,255,0)
            self.detect_counter = 0
        else:
            cv2.putText(frame,"Intrusion Alert!!!!",(0, 200),cv2.FONT_HERSHEY_PLAIN,10,(0,0,255),20)
            self.detect_counter+=1
        cv2.rectangle(overlay, (self.roi[0],self.roi[1]), (self.roi[2],self.roi[3]), self.roi_color, -1)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        self.timer.tic()
        online_im = plot_tracking(
            frame, online_tlwhs, online_ids, frame_id=self.frame_id, fps=1. / self.timer.average_time
        )
        # Show the frame in a unique window for each stream
        window_name = "YOLOv8m Tracking"
        cv2.imshow(window_name, online_im)
        # Exit on key press (applies to all streams)
        if cv2.waitKey(20) == ord('q'):
            self.done = True
        
        self.frame_id+=1

###################################################################################################
###################################################################################################
###################################################################################################

def main(args):
    """
    The main funtion
    """
    App = IntrusionMxa(args,show=True)
    App.run()

###################################################################################################

if __name__=="__main__":
    args = make_parser().parse_args()
    main(args)

# eof
