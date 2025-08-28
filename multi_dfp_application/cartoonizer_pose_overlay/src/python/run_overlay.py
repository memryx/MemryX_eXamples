import sys
import cv2 as cv
import argparse
from queue import Queue
from memryx import AsyncAccl
from apps import Cartoonizer, PoseEstimation
from PyQt5.QtWidgets import QApplication
from displayer import DisplayerWithCheckboxes
from memryx.runtime import SchedulerOptions
from threading import Thread, Event
import signal
from constant import *

# Shared flag to control thread execution
stop_flag = Event()


def signal_handler(sig, frame):
    print("\n[!] Ctrl+C received. Stopping threads...")
    stop_flag.set()


def shared_capture_loop(displayer, src, queues, stop_flag):
    frame_id = 0

    cap = cv.VideoCapture(src)
    is_cam = isinstance(src, int) or ((isinstance(src, str) and src.startswith("/dev/video")))

    while not stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        displayer.update_buffer(frame_id, RAW_FRAME_NAME, frame)

        for q in queues:
            q.put((frame_id, frame.copy()))

        frame_id += 1

    cap.release()

    if not is_cam:
        stop_flag.set()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cartoonizer and Pose Estimation demo with Multi-DFP"
    )
    parser.add_argument(
        "--cam", action="store_true", help="Use webcam instead of video"
    )
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument(
        "--frame_limit",
        "-f",
        type=int,
        default=24,
        help="Number of frames to process before swapping out.",
    )
    parser.add_argument(
        "--dfp_cartoon",
        type=str,
        default="../../models/Facial_cartoonizer_512_512_3_onnx.dfp",
        help="Path to the compiled DFP file for Cartoonizer",
    )
    parser.add_argument(
        "--dfp_pose",
        type=str,
        default="../../models/YOLO_v8_small_pose_640_640_3_onnx.dfp",
        help="Path to the compiled DFP file for Pose Estimation",
    )
    parser.add_argument(
        "--post",
        "-post",
        type=str,
        default="../../models/YOLO_v8_small_pose_640_640_3_onnx_post.onnx",
        help="Path to the ONNX post-processing model for Pose Estimation",
    )
    parser.add_argument(
        "--display_fps",
        type=int,
        default=30,
        help="Display frames per second",
    )
    parser.add_argument(
        "--frame_queue_size",
        type=int,
        default=1,
        help="Size of the frame processing queue",
    )
    return parser.parse_args()


def main():

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    args = parse_args()

    # Set up input source
    if args.cam:
        input_source = 0  # cam 0
        src_is_cam = True
    elif args.video:
        input_source = args.video
        src_is_cam = False
    else:
        print(" Please specify either --cam or --video <path>")
        sys.exit(1)

    # Initialize QApplication and Displayer
    dfp_names = [CARTOONIZER_NAME, POSE_ESTIMATION_NAME]  # will be showed in checkboxes
    app = QApplication(sys.argv)
    displayer = DisplayerWithCheckboxes(dfp_names, args.display_fps)
    displayer.show()

    # Create queues for cartoonizer and pose estimation
    queue_cartoonizer = Queue(maxsize=args.frame_queue_size)
    queue_pose = Queue(maxsize=args.frame_queue_size)

    # Start the capture process
    capture_thread = Thread(
        target=shared_capture_loop,
        args=(displayer, input_source, [queue_cartoonizer, queue_pose], stop_flag),
    )

    # Setup SchedulerOptions for AsyncAccl
    sche_opts = SchedulerOptions()
    sche_opts.frame_limit = args.frame_limit
    sche_opts.ifmap_queue_size = 22
    sche_opts.ofmap_queue_size = 30

    # Initialize AsyncAccl for pose estimation and cartoonizer
    accl_cartoonizer = AsyncAccl(args.dfp_cartoon, scheduler_options=sche_opts)
    accl_pose = AsyncAccl(args.dfp_pose, scheduler_options=sche_opts)
    accl_pose.set_postprocessing_model(args.post, model_idx=0)

    Cartoonizer(
        queue_cartoonizer,
        accl_cartoonizer,
        displayer,
        CARTOONIZER_NAME,
        src_is_cam,
        stop_flag,
    )

    PoseEstimation(
        queue_pose,
        accl_pose,
        displayer,
        POSE_ESTIMATION_NAME,
        src_is_cam,
        stop_flag,
    )

    print("[MAIN] Starting capture process and inference threads...")
    capture_thread.start()

    try:
        sys.exit(app.exec_())  # start the Qt event loop
    finally:
        print("[MAIN] App shutting down, signaling threads to stop...")
        stop_flag.set()
        accl_cartoonizer.wait()
        accl_pose.wait()

        capture_thread.join()

        app.quit()


if __name__ == "__main__":
    main()
