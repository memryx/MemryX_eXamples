import os, sys
from pathlib import Path
import argparse

try:
    import memryx
except ImportError:
    mix_home = os.getenv("MIX_HOME")
    if not mix_home:
        print("Install MemryX SDK or clone MIX and source setup_env.sh")
        exit(1)
    sys.path.append(mix_home)

import queue
import cv2 as cv
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from yolox import YoloXM
from memryx import AsyncAccl

class YoloApp:
    def __init__(self, model, cap, show=False, mirror=False, src_is_cam=True):
        self.model = model
        self.cap = cap
        self.show = show
        self.mirror = mirror
        self.src_is_cam = src_is_cam

        self.input_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.input_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        fps = int(cap.get(cv.CAP_PROP_FPS))

        self.capture_queue = queue.Queue(maxsize=5)
        self.frame_count = 0

        self.color_wheel = np.array(np.random.random([20,3])*255).astype(np.int32)

    def generate_frame(self):
        while True:
            ok, img = self.cap.read()
            if not ok:
                print('EOF')
                return None

            if self.src_is_cam and self.capture_queue.full():
                # drop frame
                continue
            else:
                if self.mirror:
                    img = cv.flip(img, 1)
                self.capture_queue.put(img)
                return self.model.preprocess(img)

    def process_model_output(self, *fmaps):
        frame = self.capture_queue.get()
        dets = self.model.postprocess(fmaps)
        self.draw(dets, frame)
        cv.imshow('dets', frame)
        self.frame_count += 1
        return frame

    def draw(self, dets, frame):
        for d in dets:
            l,t,r,b = d['bbox']
            color = tuple([int(c) for c in self.color_wheel[d['class_idx']%20]])

            frame = cv.rectangle(frame, (l,t), (r,b), color, 2)
            frame = cv.rectangle(frame, (l,t-18), (r,t), color, -1)

            frame = cv.putText(frame, d['class'], (l+2,t-5),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            shading=True
            if shading:
                crop_t, crop_b = max(0,t), min(frame.shape[0],b)
                crop_l, crop_r = max(0,l), min(frame.shape[1],r)
                box_crop = frame[crop_t:crop_b, crop_l:crop_r]
                tbox = np.ones(box_crop.shape)*np.array(color).reshape(1,1,3)
                res = cv.addWeighted(box_crop, 0.8, tbox.astype(np.uint8), 0.2, 1.0)
                if res is not None:
                    frame[crop_t:crop_b, crop_l:crop_r] = res

        #txt = '{}'.format(self.model.name)
        #frame = cv.putText(frame, txt, (50,50), cv.FONT_HERSHEY_SIMPLEX, 1,
        #       (255,0,0), 2)

        if self.show:
            cv.imshow('dets', frame)
            if cv.waitKey(1) == ord('q'):
                cv.destroyAllWindows()
                self.cap.release()
                exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOX video inference")
    parser.add_argument('-d','--dfp_path', type=str, default='models/yolox_m.dfp', 
                        help='Path to the DFP file')
    parser.add_argument('--video-source', type=str, default='/dev/video0', 
                        help='Path to video source or camera device (default is /dev/video0)')
    return parser.parse_args()

if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()

    # Open the video capture device or file
    if "/dev/video" in str(args.video_source):
        src_is_cam = True
    else:
        src_is_cam = False
    cap = cv.VideoCapture(args.video_source)
    input_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    input_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv.CAP_PROP_FPS))

    # Initialize the YoloX model and pass the dfp_path from the arguments
    model = YoloXM(img_size=(input_width, input_height), dfp_path=args.dfp_path)

    # Initialize the YoloApp and the AsyncAccl object
    app = YoloApp(model, cap, show=True, src_is_cam=src_is_cam)
    accl = AsyncAccl(model.dfp_path)

    # Connect the input and output callbacks
    accl.connect_input(app.generate_frame)
    accl.connect_output(app.process_model_output)

    # Start processing
    accl.wait()

    print("Inference completed.")
