import os
import sys
from pathlib import Path
import cv2 as cv
import numpy as np
import queue
import time
import threading
from collections import deque
import argparse
import warnings
from queue import Queue

from lib.accl import AsyncAccl

class App:
    def __init__(self, cam, show=True, save=False, scale=1.0, time_code=True, mirror=True, src_is_cam=True):
        self.cam = cam
        self.input_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT)*scale)
        self.input_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH)*scale)
        self.fps = int(cam.get(cv.CAP_PROP_FPS))
        self.capture_queue = Queue(maxsize=5)
        self.frame_times = deque(maxlen=30)
        self.show = show
        self.save = save
        self.time_code = time_code
        self.mirror = mirror
        self.writer = None
        self.prev_t = None
        self.frame_count = 0
        self.src_is_cam = src_is_cam
        if save:
            self.writer = self.setup_writer(self.input_height, self.input_width, self.fps)

    def _free(self, cap, writer=None):
        cap.release()
        if writer is not None:
            writer.release()
        time.sleep(0.5) # a small delay allows a clean exit

    def get_frame(self):
        while True:
            ok, frame = self.cam.read()
            if not ok:
                print("EOF")
                return None
            if self.src_is_cam and self.capture_queue.full():
                # drop frame
                continue
            else:
                if self.mirror:
                    frame = cv.flip(frame, 1)
                self.capture_queue.put(frame)
                return np.transpose(self.preprocess(frame), (2,3,0,1))

    def preprocess(self, img):
        arr = np.array(cv.resize(img, (512, 512))).astype(np.float32)
        arr = arr/127.5 - 1
        # add batch dimension and change to channel first format 
        # Ensure that the input shape to the accelerator matches the input shapes of the ONNX model.
        arr = np.expand_dims(arr, 0)
        arr = np.transpose(arr, (0,3,1,2))
        return arr

    def postprocess(self, frame, original_shape):
        frame = (frame + 1) * 127.5
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = cv.resize(frame, original_shape)
        return frame

    def setup_writer(self, h, w, fps):
        fname = 'result.mp4'
        print(f"Saving output to {fname}")
        writer = cv.VideoWriter(fname, cv.VideoWriter.fourcc(*'mp4v'), fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError("Failed to open writer")
        return writer

    def process_model_output(self, *ofmaps):
        img = self.postprocess(ofmaps[0], (self.input_width, self.input_height))
        self.display(img)
        self.write(img)
        self.capture_queue.get() # remove orig frame to free memory
        return img

    def display(self, img):
        if not self.show:
            return
        cv.imshow("Facial cartoonizer demo", img)
        if cv.waitKey(1) == ord('q'):
            self._free(self.cam, self.writer)
            exit(1)

    def write(self, img):
        if self.save:
            self.writer.write(img)

def parse_args():
    parser = argparse.ArgumentParser(description="Facial cartoonizer demo")
    parser.add_argument('-d', '--model_or_dfp', default='models/facial-cartoonizer_512.dfp', help="Path to the model or the corresponding DFP")
    parser.add_argument('-f', '--input_file', default=None, metavar="", help="Path to the video file to be cartoonized")
    parser.add_argument('-id', '--vid_cap_id', default=0, type=int, metavar="", help="Integer ID/index of the video capturing device to open and use")
    parser.add_argument('-s', '--save', default=False, action='store_true', help="Write cartoonized video to disk")
    parser.add_argument('--show', default=True, action ='store_true', help="Display cartoonized video")
    parser.add_argument('-x', '--scale', default=1.0, type=float, help="Scale the final output by this amount")
    args = parser.parse_args()
    return args

def setup_data(video_path):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video source")
    return cap

def main():
    args = parse_args()
    dfp = args.model_or_dfp

    if args.input_file is None:
        video_src = args.vid_cap_id
        src_is_cam = True
    else:
        video_src = args.input_file
        src_is_cam = False

    cap = setup_data(video_src)

    app = App(cap, show=args.show, save=args.save, scale=args.scale, src_is_cam=src_is_cam)
    accl = AsyncAccl(dfp=dfp)
    accl.connect_input(app.get_frame)
    accl.connect_output(app.process_model_output)
    accl.wait()

if __name__ == '__main__':
    main()
