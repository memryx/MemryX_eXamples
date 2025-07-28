import sys
import os
from pathlib import Path
from queue import Queue
import time
from collections import Counter, deque
import logging

import cv2 as cv
import numpy as np

from memryx import AsyncAccl  # Import AsyncAccl from MemryX SDK

# App class to handle camera feed, emotion recognition, and display
class App:
    def __init__(self, cam, smooth_duration=7, mirror=False, **kwargs):
        self.cam = cam  # Video capture (webcam)
        self.input_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))  # Get camera frame height
        self.input_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))  # Get camera frame width
        self.mirror = mirror  # Mirror the camera feed if required
        self.cap_queue = Queue()  # Queue to store frames
        # Emotion class labels mapped to their index
        self.idx_to_class = {
            0: 'Anger',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happiness',
            4: 'Neutral',
            5: 'Sadness',
            6: 'Surprise'
        }
        self.emoji_dir = Path(__file__).resolve().parent / 'emojis'  # Directory for emoji images
        # Load emojis corresponding to emotion classes
        self.idx_to_emoji = {
            0: self._load_emoji(self.emoji_dir/'1F92C_color.png'),
            1: self._load_emoji(self.emoji_dir/'1F92E_color.png'),
            2: self._load_emoji(self.emoji_dir/'1F628_color.png'),
            3: self._load_emoji(self.emoji_dir/'1F604_color.png'),
            4: self._load_emoji(self.emoji_dir/'1F610_color.png'),
            5: self._load_emoji(self.emoji_dir/'1F622_color.png'),
            6: self._load_emoji(self.emoji_dir/'1F62F_color.png'),
        }
        # Set margins for the image to focus on the face
        self.horz_margin = 150
        self.vert_margin = 20

        self.emotion_queue = deque()  # Queue to store recent emotion indices for smoothing
        self.emotion_ctr = Counter()  # Counter for emotion occurrences in the queue
        self.emotion_duration = smooth_duration  # Duration for emotion smoothing

    # Load emoji image from file and resize it
    def _load_emoji(self, path, shape=(128, 128)):
        emoji = cv.imread(str(path), cv.IMREAD_UNCHANGED)  # Read image with transparency
        return cv.resize(emoji, shape, cv.INTER_CUBIC)  # Resize to specified shape

    # Generate a preprocessed frame for emotion recognition model
    def generate_frame(self):
        ok, frame = self.cam.read()  # Capture a frame from the camera
        if not ok:
            return None  # Handle end of video stream
        if self.mirror:
            frame = cv.flip(frame, 1)  # Mirror the frame if required
        self.cap_queue.put(frame)  # Put the original frame in the queue
        return self.preprocess(frame)  # Return the preprocessed frame

    # Process model output and update the display with detected emotion
    def process_model_output(self, *ofmaps):
        # Postprocess model output to get emotion index
        emotion_idx = self.postprocess(ofmaps[0], (self.input_width, self.input_height))
        logging.debug(self.idx_to_class[emotion_idx])  # Log the detected emotion
        img = self.cap_queue.get()  # Get the original frame from the queue
        out = self.draw(img, emotion_idx)  # Draw the emoji corresponding to the emotion
        self.show(out)  # Display the updated frame
        return out

    # Smooth the emotion detection result over time
    def _smooth_emotion(self, emotion_idx):
        if len(self.emotion_queue) == self.emotion_duration:
            oldest = self.emotion_queue.popleft()  # Remove the oldest emotion
            self.emotion_ctr[oldest] -= 1  # Decrement count for the oldest emotion
            if self.emotion_ctr[oldest] == 0:
                del self.emotion_ctr[oldest]  # Remove it from the counter if count reaches 0
        self.emotion_queue.append(emotion_idx)  # Add the current emotion to the queue
        self.emotion_ctr[emotion_idx] += 1  # Increment its count in the counter
        # Get the most frequent emotion in the queue
        argmax = None
        max_count = 0
        for idx, count in self.emotion_ctr.items():
            if count > max_count:
                max_count = count
                argmax = idx
        return argmax

    # Preprocess the frame before passing it to the model
    def preprocess(self, img):
        if self.mirror:
            img = cv.flip(img, 1)  # Mirror the image if required
        # Crop the image to focus on the center of the face
        center = img[self.vert_margin : -self.vert_margin, self.horz_margin : -self.horz_margin, :]
        arr = np.array(cv.resize(center, (224, 224))).astype(np.float32)  # Resize to model input shape
        # Subtract mean values for normalization
        arr[..., 0] -= 103.939
        arr[..., 1] -= 116.779
        arr[..., 2] -= 123.68
        return arr

    # Draw the emoji corresponding to the detected emotion on the image
    def draw(self, img, idx):
        emoji = self.idx_to_emoji[idx]  # Get the emoji for the detected emotion
        h, w, c = emoji.shape
        ratio = 0.75  # Scale factor for the emoji
        # Calculate alpha transparency for blending
        alpha = 1 - np.expand_dims(emoji[:, :, -1]/255.0, -1)
        alpha = np.repeat(alpha, 3, axis=-1).astype(np.float32)
        fg = cv.multiply(alpha, img[:h, :w, :].astype(np.float32))  # Foreground blending
        bg = cv.multiply((1 - alpha), emoji[:, :, :-1].astype(np.float32))  # Background blending
        img[:h, :w, :] = cv.add(fg, bg)  # Combine foreground and background
        return img

    # Display the image with emotion overlay
    def show(self, img):
        cv.imshow('Emotions', img)  # Show the image in a window
        if cv.waitKey(1) == ord('q'):  # Exit on 'q' key press
            self.cam.release()  # Release the camera resource
            cv.destroyAllWindows()  # Close OpenCV windows
            os._exit(0)

    # Postprocess the logits from the model to get emotion index
    def postprocess(self, logits, original_shape):
        emotion_idx = np.argmax(np.squeeze(logits))  # Get the index of the highest logit
        emotion_idx = self._smooth_emotion(emotion_idx)  # Smooth the emotion detection result
        return emotion_idx

# Function to run the MemryX accelerator with the app
def run_mxa(dfp):
    accl = AsyncAccl(dfp)  # Initialize AsyncAccl with the DFP file
    accl.connect_input(app.generate_frame)  # Connect the input function for frame generation
    accl.connect_output(app.process_model_output)  # Connect the output function for model processing
    accl.wait()  # Wait for asynchronous processing

# Main block to run the application
if __name__ == '__main__':
    cam = cv.VideoCapture('/dev/video0')  # Open video capture (webcam)
    parent_path = Path(__file__).resolve().parent  # Get the parent directory path
    app = App(cam, mirror=True)  # Initialize the application
    dfp = parent_path / 'mobilenet_7.dfp'  # Path to the DFP file
    run_mxa(dfp)  # Run the application with MemryX
