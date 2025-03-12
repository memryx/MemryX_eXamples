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
from memryx import AsyncAccl
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

    def __init__(self, video_path, dfp_path, postmodel_path, show=True, ):
        """
        The initialization function.
        """

        # Controls
        self.show = show
        self.done = False
        self.dfp_path = dfp_path
        self.postmodel_path = postmodel_path

        # Stream-related containers
        # CV and Queues
        self.num_frames = 0
        self.cap_queue = Queue(maxsize=4)
        self.dets_queue = Queue(maxsize=5)
        if "/dev/video" in str(video_path):
            self.src_is_cam = True
        else:
            self.src_is_cam = False
        self.vidcap = cv2.VideoCapture(video_path) 
        self.dims = ( int(self.vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                int(self.vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)) )
        self.color_wheel = np.array(np.random.random([20,3])*255).astype(np.int32)

        # Model
        self.model = YoloModel(stream_img_size=(self.dims[1],self.dims[0],3))

        # Timing and FPS
        self.dt_index = 0
        self.frame_end_time = 0
        self.fps = 0
        self.dt_array = np.zeros([30])


        # Display and Save Thread
        # Runnting the display and save as a thread enhance the pipeline performance
        # Otherwise, the display_save method can be called from the output method
        self.display_thread = Thread(target=self.display,args=(), daemon=True)

    ###############################################################################
    # Run inference function to init accl and start ###############################
    ###############################################################################

    def run(self):
        """
        The function that starts the inference on the MXA.
        """
        print("dfp path = ", self.dfp_path)
        accl = AsyncAccl(dfp=self.dfp_path)
        print("YOLOv7-Tiny inference on MX3 started")
        accl.set_postprocessing_model(self.postmodel_path, model_idx=0)

        self.display_thread.start()

        start_time = time.time()

        # Connect the input and output functions and let the accl run
       
        accl.connect_input(self.capture_and_preprocess)
        accl.connect_output(self.postprocess)
        accl.wait()
        self.done = True

        # Join the display thread
        self.display_thread.join()

    ###############################################################################
    # Input and Output functions ##################################################
    ###############################################################################
    # Capture frames for streams and pre process
    def capture_and_preprocess(self):
        """
        Captures a frame for the video device and pre-processes it.
        """

        while True:

            got_frame, frame = self.vidcap.read()

            if not got_frame:
                return None


            if self.src_is_cam and self.cap_queue.full():
                # drop the frame and try again
                continue
            else:
                self.num_frames += 1
                
                # Put the frame in the cap_queue to be overlayed later
                self.cap_queue.put(frame)
                
                # Preporcess frame
                frame = self.model.preprocess(frame)
                return frame
        
    ###############################################################################
    # Post process the output from MXA
    def postprocess(self, *mxa_output):
        """
        Post-process the MXA output.
        """
         # Post-process the MXA ouptut
        dets = self.model.postprocess(mxa_output)

        # Push the results to the queue to be used by the display_save thread
        self.dets_queue.put(dets)

        # Calculate current FPS
        self.dt_array[self.dt_index] = time.time() - self.frame_end_time
        self.dt_index +=1
        
        if self.dt_index % 15 == 0:
            self.fps = 1 / np.average(self.dt_array)

            if self.dt_index >= 30:
                self.dt_index = 0
        
        self.frame_end_time = time.time()

    ###############################################################################
    # Display the output and show if opted in
    def display(self):
        """
        Continuously draws boxes over the original image for each stream and displays them in separate windows.
        """
        while not self.done:
             
            # Get the frame from and the dets from the relevant queues
            frame = self.cap_queue.get()
            dets = self.dets_queue.get()
            self.cap_queue.task_done()
            self.dets_queue.task_done()

            # Draw the OD boxes
            for d in dets:
                l,t,r,b = d['bbox']
                color = tuple([int(c) for c in self.color_wheel[d['class_idx']%20]])
                frame = cv2.rectangle(frame, (l,t), (r,b), color, 2) 
                frame = cv2.rectangle(frame, (l,t-18), (r,t), color, -1) 
                frame = cv2.putText(frame, d['class'], (l+2,t-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            #if self.fps > 1:
            #    txt = f"{self.fps:.1f} FPS"
            #else:
            #    txt = f""
            #frame = cv2.putText(frame, txt, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0), 2) 

            # Show the frame
            if self.show:

                cv2.imshow('YOLOv7-Tiny on MemryX MXA', frame)

                # Exit on a key press
                if cv2.waitKey(1) == ord('q'):
                    self.done = True
                    cv2.destroyAllWindows()
                    self.vidcap.release()
                    exit(1)

###############################################################################
# Main function of the application ############################################
###############################################################################

def main(args):
    """
    The main funtion
    """

    yolo7_inf = Yolo7Mxa(video_path = args.video_path, dfp_path=args.dfp, postmodel_path=args.postmodel, show=args.show,)
    yolo7_inf.run()

###############################################################################

if __name__=="__main__":
 
    # The args parser to parse input paths
    parser = argparse.ArgumentParser(description="\033[34mRun MX3 real-time inference with options for DFP file and post model file path.\033[0m")
    
    parser.add_argument('-d', '--dfp', 
                        type=str, 
                        default="../../models/YOLO_v7_tiny_416_416_3_onnx.dfp", 
                        help="Specify the path to the compiled DFP file. Default is 'models/YOLO_v7_tiny_416_416_3_onnx.dfp'.")
    
    parser.add_argument('-m', '--postmodel', 
                        type=str, 
                        default="../../models/YOLO_v7_tiny_416_416_3_onnx_post.onnx", 
                        help="Specify the path to the post-processing model. Default is 'models/YOLO_v7_tiny_416_416_3_onnx_post.onnx'.")

    parser.add_argument('--video_path',  dest="video_path", 
                        action="store", 
                        default='/dev/video0',
                        help="the path to video file to run inference on. Use '/dev/video0' for a webcam \n (Default: '/dev/video0')")

    parser.add_argument('--no_display', dest="show", 
                        action="store_false", 
                        default=True,
                        help="Optionally turn off the video display")

    args = parser.parse_args()

    # Call the main function
    main(args)
