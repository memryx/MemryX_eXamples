"""
============
Information:
============
Project: YOLOv7-tiny example code on MXA
File Name: run_on_mxa.py

============
Description:
============
A script to show how to use the MultiStreamAcclerator API to perform a real-time inference
on MX3 using YOLOv7-tiny model.
"""

###############################################################################
# Import necessary libraries ##################################################
###############################################################################

import time
import argparse
import numpy as np
import cv2
from queue import Queue, Full
from threading import Thread
from matplotlib import pyplot as plt
from memryx import MultiStreamAsyncAccl
from yolov7 import YoloV7Tiny as YoloModel


###############################################################################
# Class with necessary methods to run the application #########################
###############################################################################

class Yolo7Mxa:
    """
    A demo app to run YOLOv7 on the MemryX MXA.
    """

    ###############################################################################
    # Class constructor ###########################################################
    ###############################################################################

    def __init__(self, video_paths, dfp_path, postmodel_path, show=True, ):
        """
        The initialization function.
        """

        # Controls
        self.show = show
        self.done = False
        self.dfp_path = dfp_path
        self.postmodel_path = postmodel_path

        self.num_streams = len(video_paths)

        # Stream-related containers
        self.streams = []
        self.streams_idx = [True] * self.num_streams
        self.stream_window = [False] * self.num_streams
        self.cap_queue = {i: Queue(maxsize=4) for i in range(self.num_streams)}
        self.dets_queue = {i: Queue(maxsize=5) for i in range(self.num_streams)}
        self.outputs = {i: [] for i in range(self.num_streams)}
        self.dims = {}
        self.color_wheel = {}
        self.model = {}

        # Timing and FPS related
        self.dt_index = {i: 0 for i in range(self.num_streams)}
        self.frame_end_time = {i: 0 for i in range(self.num_streams)}
        self.fps = {i: 0 for i in range(self.num_streams)}
        self.dt_array = {i: np.zeros(30) for i in range(self.num_streams)}
        self.writer = {i: None for i in range(self.num_streams)}
        self.srcs_are_cams = {i: True for i in range(self.num_streams)}

        # Initialize video captures, models, and dimensions for each streams
        for i, video_path in enumerate(video_paths):
            if "/dev/video" in video_path:
                self.srcs_are_cams[i] = True
            else:
                self.srcs_are_cams[i] = False

            vidcap = cv2.VideoCapture(video_path)
            self.streams.append(vidcap)

            self.dims[i] = (int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.color_wheel[i] = np.random.randint(0, 255, (20, 3)).astype(np.int32)

            # Initialize the model with the stream dimensions
            self.model[i] = YoloModel(stream_img_size=(self.dims[i][1], self.dims[i][0], 3))
        
        self.display_thread = Thread(target=self.display)

    ###############################################################################
    # Run inference function to init accl and start ###############################
    ###############################################################################

    def run(self):
        """
        The function that starts the inference on the MXA.
        """
        print("dfp path = ", self.dfp_path)
        accl = MultiStreamAsyncAccl(dfp=self.dfp_path)
        print("YOLOv7-Tiny inference on MX3 started")
        accl.set_postprocessing_model(self.postmodel_path, model_idx=0)

        self.display_thread.start()

        start_time = time.time()

        # Connect the input and output functions and let the accl run
        accl.connect_streams(self.capture_and_preprocess, self.postprocess, self.num_streams)
        accl.wait()

        self.done = True

        # Join the display thread
        self.display_thread.join()

    ###############################################################################
    # Input and Output functions ##################################################
    ###############################################################################
    # Capture frames for streams and pre process
    def capture_and_preprocess(self, stream_idx):
        """
        Captures a frame for the video device and pre-processes it.
        """
        # if self.srcs_are_cams[stream_idx]:
        while True:
            got_frame, frame = self.streams[stream_idx].read()

            if not got_frame or self.done:
                self.streams_idx[stream_idx] = False
                return None

            if self.srcs_are_cams[stream_idx] and self.cap_queue[stream_idx].full():
                # drop frame
                continue
            else:
                try:
                    # Put the frame in the cap_queue to be processed later
                    self.cap_queue[stream_idx].put(frame, timeout=2)

                    # Pre-process the frame using the corresponding model
                    frame = self.model[stream_idx].preprocess(frame)
                    return frame

                except Full:
                    print('Dropped frame')
                    continue

    ###############################################################################
    # Post process the output from MXA
    def postprocess(self, stream_idx, *mxa_output):
        """
        Post-process the MXA output.
        """
        dets = self.model[stream_idx].postprocess(mxa_output)

        # Push the detection results to the queue
        self.dets_queue[stream_idx].put(dets)

        # Calculate the FPS
        self.dt_array[stream_idx][self.dt_index[stream_idx]] = time.time() - self.frame_end_time[stream_idx]
        self.dt_index[stream_idx] += 1

        if self.dt_index[stream_idx] % 15 == 0:
            self.fps[stream_idx] = 1 / np.average(self.dt_array[stream_idx])

        if self.dt_index[stream_idx] >= 30:
            self.dt_index[stream_idx] = 0

        self.frame_end_time[stream_idx] = time.time()

    ###############################################################################
    # Display the output and show if opted in
    def display(self):
        """
        Continuously draws boxes over the original image for each stream and displays them in separate windows.
        """
        while not self.done:
            # Iterate over each stream to handle multiple displays
            for stream_idx in range(self.num_streams):
                # Check if the queues for frames and detections have data
                if not self.cap_queue[stream_idx].empty() and not self.dets_queue[stream_idx].empty():
                    frame = self.cap_queue[stream_idx].get()
                    dets = self.dets_queue[stream_idx].get()

                    self.cap_queue[stream_idx].task_done()
                    self.dets_queue[stream_idx].task_done()

                    # Draw detection boxes on the frame
                    for d in dets:
                        l, t, r, b = d['bbox']
                        color = tuple(int(c) for c in self.color_wheel[stream_idx][d['class_idx'] % 20])
                        frame = cv2.rectangle(frame, (l, t), (r, b), color, 2)
                        frame = cv2.rectangle(frame, (l, t - 18), (r, t), color, -1)
                        frame = cv2.putText(frame, d['class'], (l + 2, t - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # Add FPS information to the frame
                    fps_text = f"{self.model[stream_idx].name} - {self.fps[stream_idx]:.1f} FPS" if self.fps[stream_idx] > 1 else self.model[stream_idx].name
                    frame = cv2.putText(frame, fps_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    # Show the frame in a unique window for each stream
                    if self.show:
                        window_name = f"Stream {stream_idx} - YOLOv7-Tiny"
                        cv2.imshow(window_name, frame)

            # Exit on key press (applies to all streams)
            if cv2.waitKey(1) == ord('q'):
                self.done = True

        # When done, destroy all windows and release resources
        cv2.destroyAllWindows()
        for stream in self.streams:
            stream.release()

###############################################################################
# Main function of the application ############################################
###############################################################################

def main(args):
    """
    The main funtion
    """

    yolo7_inf = Yolo7Mxa(video_paths = args.video_paths, dfp_path=args.dfp, postmodel_path=args.postmodel, show=args.show,)
    yolo7_inf.run()

###############################################################################

if __name__=="__main__":
 
    # The args parser to parse input paths
    parser = argparse.ArgumentParser(description="\033[34mRun MX3 real-time inference with options for DFp file and post model file path.\033[0m")
    
    parser.add_argument('-d', '--dfp', 
                        type=str, 
                        default="../../models/YOLO_v7_tiny_416_416_3_onnx.dfp", 
                        help="Specify the path to the compiled DFP file. Default is 'models/YOLO_v7_tiny_416_416_3_onnx.dfp'.")
    
    parser.add_argument('-m', '--postmodel', 
                        type=str, 
                        default="../../models/YOLO_v7_tiny_416_416_3_onnx_post.onnx", 
                        help="Specify the path to the post-processing model. Default is 'models/YOLO_v7_tiny_416_416_3_onnx_post.onnx'.")

    parser.add_argument('--video_paths', nargs='+',  dest="video_paths", 
                        action="store", 
                        default=['/dev/video0'],
                        help="the path to video file to run inference on. Use '/dev/video0' for a webcam \n For multiple input videos just provide the paths with space inbetween them. (Default: '/dev/video0')")

    parser.add_argument('--no_display', dest="show", 
                        action="store_false", 
                        default=True,
                        help="Optionally turn off the video display")

    args = parser.parse_args()

    # Call the main function
    main(args)