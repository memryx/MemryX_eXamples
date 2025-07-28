import sys
import os
import argparse
from pathlib import Path
from queue import Queue
import time
import threading
from collections import namedtuple
import logging
import signal

import cv2 as cv
import numpy as np

from memryx import AsyncAccl
from face_detection.app import App as FaceApp
from emotion_recognition.app import App as EmotionApp

# FaceCropper extends FaceApp to crop faces from the frame
class FaceCropper(FaceApp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Map model outputs to specific indices
        self.model.output_map = {
            'regressor_8': 2,
            'regressor_16': 3,
            'classificator_8': 0,
            'classificator_16': 1,
        }
        self.scale = kwargs.get('scale', 1)

    # Generate the frame from face detection
    def generate_frame_face(self):
        return self.generate_frame()

    # Process the output feature maps (ofmaps) from face detection model
    def process_face(self, *ofmaps):
        dets = self.model.postprocess(*ofmaps)  # Postprocess the model's output
        frame = self.capture_queue.get()  # Get the frame from the capture queue
        if dets.size == 0:
            return frame, 0  # If no faces are detected, return original frame
        face = self._crop_faces(frame, dets)  # Crop the detected face
        return face, dets.shape[0]

    # Crop the detected face area from the frame
    def _crop_faces(self, frame, dets):
        # Only pick the first face detected
        xmin, ymin, xmax, ymax = self._get_box(dets[0])
        # Clip the coordinates to ensure they are within the frame boundaries
        xmin = np.clip(xmin, 0, frame.shape[1])
        ymin = np.clip(ymin, 0, frame.shape[0])
        xmax = np.clip(xmax, 0, frame.shape[1])
        ymin = np.clip(ymin, 0, frame.shape[0])
        if self.scale != 1:
            # Adjust face bounding box based on the scale factor
            face_height = ymax - ymin
            face_width = xmax - xmin
            p = 0.5*(self.scale - 1)
            ymin = np.clip(int(ymin - p*face_height), 0, frame.shape[0])
            ymax = np.clip(int(ymax + p*face_height), 0, frame.shape[0])
            xmin = np.clip(int(xmin - p*face_width), 0, frame.shape[1])
            xmax = np.clip(int(xmax + p*face_width), 0, frame.shape[1])
        return frame[ymin : ymax + 1, xmin : xmax + 1, :]


# EmotionWithBackground extends EmotionApp to include background handling
class EmotionWithBackground(EmotionApp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.idx_to_class[7] = 'Background'  # Add 'Background' class
        # Load an emoji for background
        self.idx_to_emoji[7] = self._load_emoji(self.emoji_dir/'1F50D_color.png')
        self.background = False  # Flag to indicate if background is being used

    # Process model output with background handling
    def process_model_output(self, *ofmaps):
        if self.background:
            # If background is detected, set emotion index to 7 (Background)
            self.background = False
            emotion_idx = 7
            emotion_idx = self._smooth_emotion(emotion_idx)  # Apply smoothing
        else:
            # Otherwise, process the model's output normally
            emotion_idx = self.postprocess(ofmaps[0], (self.input_width, self.input_height))
        logging.debug(self.idx_to_class[emotion_idx])
        img = self.cap_queue.get()
        out = self.draw(img, emotion_idx)  # Draw emotion on the frame
        return out

# App class that integrates face detection and emotion recognition
class App:
    def __init__(self, cam, smooth_duration=7, mirror=False, scale=1.2, **kwargs):
        Shape = namedtuple('Shape', ['height', 'width'])  # Create a namedtuple for shape
        # Initialize face and emotion detection apps
        self.face_app = FaceCropper(cam, model_input_shape=Shape(height=128, width=128), mirror=mirror, scale=scale)
        self.emotion_app = EmotionWithBackground(cam, smooth_duration=smooth_duration, mirror=mirror)
        self.face = None
        self.capture_queue = Queue()  # Queue to store frames
        self.runflag = True

    # Generate frame for face detection
    def generate_frame_face(self):
        frame = self.face_app.generate_frame()
        frame = np.expand_dims(frame, 0)
        if frame is None:
            print("EOF")  # Handle end of the stream
            os._exit(0)
            return None
        orig_frame = self.face_app.capture_queue.get()  # Get original frame
        self.capture_queue.put(orig_frame)  # Store the frame for later use
        self.face_app.capture_queue.put(orig_frame)
        return frame

    # Process the output from face detection model
    def process_face(self, *ofmaps):
        if len(ofmaps) == 1:
            ofmaps = ofmaps[0]
        self.face, face_count = self.face_app.process_face(*ofmaps)
        if face_count == 0:
            self.emotion_app.background = True  # Set background if no face is detected
        return self.face

    # Generate frame for emotion recognition
    def generate_frame_emotion(self):
        try:
            # Resize the face crop for emotion recognition model input
            face = cv.resize(self.face, (224, 224), interpolation=cv.INTER_CUBIC)
            face = np.expand_dims(face, 0)
        except Exception:
            self.emotion_app.background = True  # Set background if face is not available
            face = np.zeros((1,224,224,3))  # Create a blank frame if no face
        self.emotion_app.cap_queue.put(self.capture_queue.get())
        return face.astype(np.float32)

    # Process the output from the emotion recognition model
    def process_emotion(self, *ofmaps):
        out = self.emotion_app.process_model_output(*ofmaps)
        self.emotion_app.show(out)  # Show the result
        return out

# Main function to run the application with MemryX AsyncAccl
def run_mxa(dfp):
    accl = AsyncAccl(dfp)  # Initialize AsyncAccl with the DFP file
    accl.connect_input(app.generate_frame_face)  # Connect face detection input
    accl.connect_input(app.generate_frame_emotion, 1)  # Connect emotion recognition input
    accl.connect_output(app.process_face)  # Connect face detection output
    accl.connect_output(app.process_emotion, 1)  # Connect emotion recognition output
    accl.wait()  # Wait for the asynchronous processing

# Entry point for the script
if __name__ == '__main__':
    # Argument parsing for DFP file
    parser = argparse.ArgumentParser(description="Face & Emotion Detection")
    parser.add_argument('-d', '--dfp', type=str, default='../../models/models.dfp', 
                        help='Path to the compiled DFP file (default: models/models.dfp)')
    
    args = parser.parse_args()

    cam = cv.VideoCapture(0)  # Open video capture (webcam)
    parent_path = Path(__file__).resolve().parent

    app = App(cam, mirror=True)  # Initialize the application
    dfp = Path(args.dfp)  # Get DFP path from argument

    run_mxa(dfp)  # Run the application with the provided DFP

    # After shutdown, release the camera and destroy all windows
    cam.release()
    cv.destroyAllWindows()
