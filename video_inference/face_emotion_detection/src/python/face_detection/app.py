import sys
import os
import subprocess
import shutil
from pathlib import Path
from abc import ABC, abstractmethod
import time
import threading
from queue import Queue
from collections import namedtuple

import numpy as np
import cv2 as cv

# Add the current directory to system path
sys.path.append(str(Path(__file__).resolve().parent))
import model  # Import custom model module

from memryx import AsyncAccl  # Import AsyncAccl class from MemryX SDK

# App class to handle camera feed, model processing, and output display
class App:
    def __init__(self, cam, model_input_shape, mirror=False, **kwargs):
        self.cam = cam
        self.input_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))  # Get camera frame height
        self.input_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))  # Get camera frame width
        self.model_input_shape = model_input_shape
        self.capture_queue = Queue(maxsize=5)  # Queue to store captured frames
        self.mirror = mirror  # Flag for mirroring the camera feed
        self.model = model.MPFaceDetector(model_input_shape)  # Initialize face detection model

    # Function to capture a frame from the camera, preprocess it, and return it
    def generate_frame(self):
        while True:
            ok, frame = self.cam.read()  # Read frame from camera
            if not ok:
                print('EOF')  # Handle end of stream
                return None
            if self.capture_queue.full():
                # drop frame (and ONLY in this first model!)
                # dependent models shouldn't drop anything
                continue
            else:
                if self.mirror:
                    frame = cv.flip(frame, 1)  # Mirror the frame if required
                self.capture_queue.put(frame)  # Put the original frame in the queue
                out = self.model.preprocess(frame)  # Preprocess frame for the model
                return out

    # Process model output and draw detected faces on the frame
    def process_model_output(self, *ofmaps):
        dets = self.model.postprocess(*ofmaps)  # Postprocess model output
        frame = self.capture_queue.get()  # Get the original frame from the queue
        out = self.draw(frame, dets)  # Draw detection results on the frame
        self.show(out)  # Show the frame with drawn results

    # Draw bounding boxes around detected faces on the image
    def draw(self, img, dets):
        for i in range(dets.shape[0]):
            xmin, ymin, xmax, ymax = self._get_box(dets[i])  # Get bounding box coordinates
            img = cv.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 3)  # Draw rectangle
        return img

    # Get bounding box coordinates in image scale from normalized detection values
    def _get_box(self, det):
        ymin = int(det[0]*self.input_height)
        xmin = int(det[1]*self.input_width)
        ymax = int(det[2]*self.input_height)
        xmax = int(det[3]*self.input_width)
        return [xmin, ymin, xmax, ymax]

    # Display the image with detected faces in a window
    def show(self, img):
        cv.imshow('Faces', img)  # Show the image in a window titled 'Faces'
        if cv.waitKey(1) == ord('q'):  # Exit on 'q' key press
            self.cam.release()  # Release the camera resource
            cv.destroyAllWindows()  # Close all OpenCV windows
            os._exit(0)

# Function to run MemryX accelerator with the application
def run_mxa(dfp):
    accl = AsyncAccl(dfp)  # Initialize AsyncAccl with DFP file
    accl.connect_input(app.generate_frame)  # Connect the input function (frame generation)
    accl.connect_output(app.process_model_output)  # Connect the output function (model processing)
    accl.wait()  # Wait for asynchronous processing to complete

# Main execution block
if __name__ == '__main__':
    cam = cv.VideoCapture('/dev/video0')  # Open video capture (webcam)
    parent_path = Path(__file__).resolve().parent  # Get the parent directory path
    dfp = parent_path / 'face_detection_short_range.dfp'  # Path to the DFP file
    Shape = namedtuple('Shape', ['height', 'width'])  # Define a namedtuple for input shape
    app = App(cam, Shape(height=128, width=128), mirror=True)  # Initialize the application
    run_mxa(dfp)  # Run the application with MemryX

