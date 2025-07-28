# --- run.py ---
import os
import sys
from pathlib import Path
import cv2 as cv
import numpy as np
from collections import deque
import argparse
from queue import Queue
from multiprocessing import Process, Queue, Event
from memryx import AsyncAccl
from apps import Cartoonizer, PoseEstmiation
from PyQt5.QtWidgets import QApplication
from displayer import Displayer
from memryx.runtime import SchedulerOptions, ClientOptions
from threading import Thread  

def parse_args():
    parser = argparse.ArgumentParser(description="Cartoonizer and Pose Estimation demo with Multi-DFP")
    parser.add_argument('--cam', action='store_true', help="Use webcam instead of video")
    parser.add_argument('--video', type=str, help="Path to video file")
    parser.add_argument('--frame_limit', '-f', type=int, default=30, help="Number of frames to process before swapping out.")
    
    # New: separate DFPs for cartoonizer and pose
    parser.add_argument('--dfp_cartoon', type=str, default='../../models/Facial_cartoonizer_512_512_3_onnx.dfp',
                        help="Path to the compiled DFP file for Cartoonizer")
    parser.add_argument('--dfp_pose', type=str, default='../../models/YOLO_v8_small_pose_640_640_3_onnx.dfp',
                        help="Path to the compiled DFP file for Pose Estimation")
    parser.add_argument('--post', '-post', type=str, default='../../models/YOLO_v8_small_pose_640_640_3_onnx_post.onnx',
                        help="Path to the ONNX post-processing model for Pose Estimation")

    return parser.parse_args()

def shared_capture_loop(src, queue1, queue2, stop_flag):
    cap = cv.VideoCapture(src)
    is_cam = src == "/dev/video0"  # Detect if it's a webcam

    while not stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        for q in [queue1, queue2]:
            if not q.full():
                q.put(frame.copy())
    cap.release()

    # Only set the stop flag after video ends (not webcam)
    if not is_cam:
        stop_flag.set()

def run_cartoonizer(queue, dfp_path, display_thread, src_is_cam, frame_limit, stop_flag):
    sched_opts = SchedulerOptions(frame_limit, 0, False, 60, 60)
    client_opts = ClientOptions(True, 25.0)
    accl = AsyncAccl(
        dfp_path,
        use_model_shape=(True, True),
        scheduler_options=sched_opts,
        client_options=client_opts
    )
    Cartoonizer(queue, accl, display_thread, src_is_cam=src_is_cam, stop_flag=stop_flag)

def run_pose_estimation(queue, dfp_path, pose_post_model, input_shape, display_thread, src_is_cam, frame_limit, stop_flag):
    sched_opts = SchedulerOptions(frame_limit, 0, False, 60, 60)
    client_opts = ClientOptions(True, 25.0)
    accl = AsyncAccl(
        dfp_path,
        use_model_shape=(True, True),
        scheduler_options=sched_opts,
        client_options=client_opts
    )
    PoseEstmiation(queue, accl, display_thread, input_shape, mirror=True, src_is_cam=src_is_cam, stop_flag=stop_flag, post_model=pose_post_model)

def main():
    args = parse_args()

    ############### init display ###############
    app = QApplication(sys.argv)
    displayer = Displayer(num_windows=2)
    displayer.show()

    dfp_cartoon = args.dfp_cartoon
    dfp_pose = args.dfp_pose
    pose_post_model = args.post

    if args.cam:
        input_source = "/dev/video0"
        src_is_cam = True
    elif args.video:
        input_source = args.video
        src_is_cam = False
    else:
        print(" Please specify either --cam or --video <path>")
        sys.exit(1)

    queue_cartoonizer = Queue(maxsize=100)
    queue_pose = Queue(maxsize=100)
    stop_flag = Event()

    # Start capture as a separate process
    capture_proc = Process(
        target=shared_capture_loop,
        args=(input_source, queue_cartoonizer, queue_pose, stop_flag)
    )

    # Run inference in threads
    cartoon_proc = Thread(
        target=run_cartoonizer,
        args=(queue_cartoonizer, dfp_cartoon, displayer.update_left,
            src_is_cam, args.frame_limit, stop_flag)
    )

    pose_proc = Thread(
        target=run_pose_estimation,
        args=(queue_pose, dfp_pose, pose_post_model, (640, 640),
            displayer.update_right, src_is_cam, args.frame_limit, stop_flag)
    )

    print("[MAIN] Starting capture process and inference threads...")
    capture_proc.start()
    cartoon_proc.start()
    pose_proc.start()

    try:
        sys.exit(app.exec_())
    finally:
        print("[MAIN] App shutting down, signaling threads to stop...")
        stop_flag.set()
        cartoon_proc.join()
        pose_proc.join()
        if capture_proc.is_alive():
            capture_proc.terminate()

        app.quit()  # Ensures Qt exits cleanly after video ends

if __name__ == '__main__':
    main()