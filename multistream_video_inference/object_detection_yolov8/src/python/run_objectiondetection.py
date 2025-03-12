"""
============
Information:
============
Project: YOLOv8s example code on MXA
File Name: app.py

============
Description:
============
A script to show how to use the MultiStreamAcclerator API to perform a real-time inference
on MX3 using YOLOv8s model.
"""

###################################################################################################

# Imports
import time
import argparse
import numpy as np
import cv2
from queue import Queue, Full
from threading import Thread
from memryx import MultiStreamAsyncAccl
from yolov8 import YoloV8 as YoloModel

###################################################################################################

class Yolo8sMxa:
    """
    A demo app to run YOLOv8s on the MemryX MXA.
    """

###################################################################################################
    def __init__(self, video_paths, show=True):
        """
        Initialization function.
        """
        # Display control and stream initialization
        self.show = show
        self.done = False
        self.num_streams = len(video_paths)  # Number of streams

        # Stream-related containers and initialization
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

        # Initialize video captures, models, and dimensions for each stream
        for i, video_path in enumerate(video_paths):
            if "/dev/video" in video_path:
                self.srcs_are_cams[i] = True
            else:
                self.srcs_are_cams[i] = False

            vidcap = cv2.VideoCapture(video_path)
            self.streams.append(vidcap)

            # Get frame dimensions
            self.dims[i] = (int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.color_wheel[i] = np.random.randint(0, 255, (20, 3)).astype(np.int32)

            # Initialize the YOLOv8 model
            self.model[i] = YoloModel(stream_img_size=(self.dims[i][1], self.dims[i][0], 3))

        # Start display thread
        self.display_thread = Thread(target=self.display)

###################################################################################################
    def run(self):
        """
        Start inference on the MXA using multiple streams.
        """
        accl = MultiStreamAsyncAccl(dfp=self.dfp)  # Initialize the accelerator with DFP
        print("YOLOv8s inference on MX3 started")
        accl.set_postprocessing_model(self.post_model, model_idx=0)  # Set the post-processing model

        self.display_thread.start()  # Start the display thread

        start_time = time.time()

        # Connect input and output streams for the accelerator
        accl.connect_streams(self.capture_and_preprocess, self.postprocess, self.num_streams)
        accl.wait()

        self.done = True

        # Join display thread
        self.display_thread.join()

###################################################################################################
    def capture_and_preprocess(self, stream_idx):
        """
        Captures a frame for the video device and pre-processes it.
        """
        # if self.srcs_are_cams[stream_idx]:
        while True:
            got_frame, frame = self.streams[stream_idx].read()

            if not got_frame:
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
                    print('Dropped frame .. exiting')
                    return None

###################################################################################################
    def postprocess(self, stream_idx, *mxa_output):
        """
        Post-process the output from MXA.
        """
        dets = self.model[stream_idx].postprocess(mxa_output)  # Get detection results

        # Queue detection results for display
        self.dets_queue[stream_idx].put(dets)

        # Calculate FPS
        self.dt_array[stream_idx][self.dt_index[stream_idx]] = time.time() - self.frame_end_time[stream_idx]
        self.dt_index[stream_idx] += 1

        if self.dt_index[stream_idx] % 15 == 0:
            self.fps[stream_idx] = 1 / np.average(self.dt_array[stream_idx])

        if self.dt_index[stream_idx] >= 30:
            self.dt_index[stream_idx] = 0

        self.frame_end_time[stream_idx] = time.time()

###################################################################################################
    def display(self):
        """
        Displays the processed frames with detections in separate windows.
        """
        while not self.done:
            # Iterate through each stream for displaying frames
            for stream_idx in range(self.num_streams):
                if not self.cap_queue[stream_idx].empty() and not self.dets_queue[stream_idx].empty():
                    frame = self.cap_queue[stream_idx].get()
                    dets = self.dets_queue[stream_idx].get()

                    self.cap_queue[stream_idx].task_done()
                    self.dets_queue[stream_idx].task_done()

                    # Draw detection boxes
                    for d in dets:
                        x1, y1, w, h = d['bbox']
                        color = tuple(int(c) for c in self.color_wheel[stream_idx][d['class_id'] % 20])

                        # Draw bounding boxes
                        frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

                        # Add class label
                        frame = cv2.putText(frame, d['class'], (x1 + 2, y1 - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # Add FPS to frame
                    fps_text = f"{self.model[stream_idx].name} - {self.fps[stream_idx]:.1f} FPS" if self.fps[stream_idx] > 1 else self.model[stream_idx].name
                    frame = cv2.putText(frame, fps_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    # Display the frame in a unique window for each stream
                    if self.show:
                        window_name = f"Stream {stream_idx} - YOLOv8s"
                        cv2.imshow(window_name, frame)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) == ord('q'):
                self.done = True

        # Close all windows and release resources after processing
        cv2.destroyAllWindows()
        for stream in self.streams:
            stream.release()

###################################################################################################

def main(args):
    """
    Main function to start YOLOv8s inference.
    """
    # Initialize the application with video paths and display settings
    yolo8s_inf = Yolo8sMxa(video_paths=args.video_paths, show=args.show)
    yolo8s_inf.dfp = args.dfp  # Set the DFP path from arguments
    yolo8s_inf.post_model = args.post_model  # Set the post-processing model path from arguments
    yolo8s_inf.run()  # Start inference

###################################################################################################

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="\033[34mMemryX YoloV8s Demo\033[0m")
    
    # Video input paths
    parser.add_argument('--video_paths', nargs='+', dest="video_paths", 
                        action="store", 
                        default=['/dev/video0'],
                        help="Path to video files for inference. Use '/dev/video0' for webcam. (Default:'/dev/video0')")
    
    # Option to turn off display
    parser.add_argument('--no_display', dest="show", 
                        action="store_false", 
                        default=True,
                        help="Optionally turn off the video display")

    # DFP model argument
    parser.add_argument('-d', '--dfp', type=str, 
                        default='../../models/tflite/YOLO_v8_small_640_640_3_tflite.dfp', 
                        help="Path to the compiled DFP file (default: 'models/tflite/YOLO_v8_small_640_640_3_tflite.dfp')")

    # Post-processing model argument
    parser.add_argument('-p', '--post_model', type=str, 
                        default='../../models/tflite/YOLO_v8_small_640_640_3_tflite_post.tflite', 
                        help="Path to the post-processing ONNX file (default: 'models/tflite/YOLO_v8_small_640_640_3_tflite_post.tflite')")

    args = parser.parse_args()

    # Call the main function
    main(args)

# eof