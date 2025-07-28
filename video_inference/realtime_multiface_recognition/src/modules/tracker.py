from PySide6.QtCore import QThread, Signal, QObject
from modules.bytetracker import BYTETracker
import queue
from dataclasses import dataclass, field
import numpy as np
from modules.MXFace import MXFace, MockMXFace, AnnotatedFrame
from pathlib import Path
import time
from .database import FaceDatabase
from .utils import Framerate
import threading  # Import the threading module for locks
from copy import deepcopy

@dataclass
class TrackedObject:
    bbox: tuple[int, int, int, int]
    keypoints: list[tuple[int, int]]
    track_id: int
    name: str = "Unknown"
    activated: bool = True
    last_recognition: float = 0.0
    distances: list[float] = field(default_factory=list)
    embedding: np.ndarray = field(default_factory=lambda: np.zeros([128]))


@dataclass
class CompositeFrame:
    image: np.ndarray
    tracked_objects: list


def compute_iou(boxA, boxB):
    # boxA and boxB are (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)


class FaceTracker(QObject):

    frame_ready = Signal(np.ndarray)

    """
    DetectionThread: continuously pulls detections from mxface.detect_get(),
    updates the tracker and pushes unknown faces (with track_id) to be recognized.
    RecognitionThread: continuously pulls recognition results from mxface.recognize_get()
    and updates the tracker_dict with the recognized name.
    """
    def __init__(self, face_database: FaceDatabase):
        super().__init__()
        self.tracker = BYTETracker()
        #self.mxface = MockMXFace(Path('assets/models'))
        self.mxface = MXFace(Path('../models'))
        self.tracker_dict = {}  # Mapping from track_id to TrackedObject
        self.tracker_dict_lock = threading.Lock()  # Lock for tracker_dict
        self.current_frame = AnnotatedFrame(np.zeros([1920, 1080, 3]))
        self.composite_queue = queue.Queue(maxsize=1)
        self.database = face_database
        
        # Create worker threads for detection and recognition
        self.detection_thread = DetectionThread(self)
        self.recognition_thread = RecognitionThread(self)

    def start(self):
        self.detection_thread.start()
        self.recognition_thread.start()

    def stop(self):
        self.detection_thread.stop()
        self.recognition_thread.stop()
        self.detection_thread.wait()
        self.recognition_thread.wait()
        self.mxface.stop()

    def detect(self, frame):
        try:
            self.mxface.detect_put(frame, block=False)
        except queue.Full:
            pass

    def get_activated_tracker_objects(self) -> list:
        with self.tracker_dict_lock:
            return [deepcopy(obj) for obj in self.tracker_dict.values() if obj.activated]


class DetectionThread(QThread):
    """
    This thread continuously calls mxface.detect_get() to get new annotated frames.
    It then updates the tracker using the detected bounding boxes, marks existing objects as inactive,
    and for each new unknown track, extracts the face and pushes a tuple (track_id, face)
    to mxface.recognize_put() for further processing.
    It also pushes a CompositeFrame (current image and activated objects) into a composite_queue.
    """
    def __init__(self, face_tracker):
        super().__init__()
        self.face_tracker = face_tracker
        self.stop_threads = False
        self.refresh_interval = 1
        self.framerate = Framerate()

    def _update_detections(self):

        try:
            annotated_frame = self.face_tracker.mxface.detect_get(timeout=0.033)
            self.face_tracker.current_frame = annotated_frame
            self.framerate.update()
        except queue.Empty:
            return

        # Mark all current tracked objects as not active (protected by lock)
        with self.face_tracker.tracker_dict_lock:
            for tracked_object in self.face_tracker.tracker_dict.values():
                tracked_object.activated = False

            current_time = time.time()
            if annotated_frame.num_detected_faces == 0:
                return

            # Build detections array expected by BYTETracker
            dets = []
            for bbox, score in zip(annotated_frame.boxes, annotated_frame.scores):
                x, y, w, h = bbox
                dets.append(np.array([x, y, x+w, y+h, score, 0]))
            dets = np.array(dets, dtype=np.float32)

            # Update tracker with the new detections
            for tracklet in self.face_tracker.tracker.update(dets, None):
                x1, y1, x2, y2, track_id, _, _ = tracklet.astype(int)
                keypoints = self._get_keypoints((x1, y1, x2, y2), annotated_frame)

                # For an existing track, update bbox and activate it.
                if track_id in self.face_tracker.tracker_dict:
                    tracked_obj = self.face_tracker.tracker_dict[track_id]
                    tracked_obj.bbox = (x1, y1, x2, y2)
                    tracked_obj.keypoints = keypoints
                    tracked_obj.activated = True

                    # Refresh active track if refresh_interval elapsed.
                    if current_time - tracked_obj.last_recognition > self.refresh_interval:
                        try:
                            self.face_tracker.mxface.recognize_put(
                                (track_id, annotated_frame.image, (x1, y1, x2, y2), (keypoints[0], keypoints[1])), block=False)
                        except queue.Full:
                            pass
                else:
                    # New track: create a new tracked object and request recognition immediately.
                    new_obj = TrackedObject(
                        bbox=(x1, y1, x2, y2), 
                        keypoints=keypoints, 
                        track_id=track_id, 
                        last_recognition=current_time
                    )
                    self.face_tracker.tracker_dict[track_id] = new_obj
                    try:
                        self.face_tracker.mxface.recognize_put(
                            (track_id, annotated_frame.image, (x1, y1, x2, y2), (keypoints[0], keypoints[1])), block=False)
                    except queue.Full:
                        pass


    def run(self):
        while not self.stop_threads:
            self._update_detections()
            if self.face_tracker.current_frame:
                self.face_tracker.frame_ready.emit(self.face_tracker.current_frame.image)

    def stop(self):
        self.stop_threads = True

    def _get_keypoints(self, track_box, annotated_frame):
        """Re-associate the tracked box with the detected box to extract keypoints."""
        best_iou = 0
        best_idx = None

        # Loop over detections from annotated_frame
        for idx, det_box in enumerate(annotated_frame.boxes):
            # Convert detection box from (x, y, w, h) to (x1, y1, x2, y2)
            det_box_converted = (det_box[0], det_box[1], det_box[0] + det_box[2], det_box[1] + det_box[3])
            iou = compute_iou(track_box, det_box_converted)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        return annotated_frame.keypoints[best_idx]


class RecognitionThread(QThread):
    """
    This thread continuously calls mxface.recognize_get() to retrieve recognition results.
    Each result is expected to be a tuple (track_id, recognized_name). The thread then updates
    the corresponding tracked object with the new recognized name.
    """
    def __init__(self, face_tracker):
        super().__init__()
        self.face_tracker = face_tracker
        self.stop_threads = False
        self.framerate = Framerate()

    def run(self):
        while not self.stop_threads:
            self.framerate.update()
            self._run()

    def _run(self):
        try:
            # Expect recognition results as (track_id, embedding)
            track_id, embedding = self.face_tracker.mxface.recognize_get(timeout=0.1)
        except queue.Empty:
            return

        with self.face_tracker.tracker_dict_lock:
            if track_id not in self.face_tracker.tracker_dict:
                return

            name, distances = self.face_tracker.database.find(embedding)
            tracked_obj = self.face_tracker.tracker_dict[track_id]
            tracked_obj.embedding = embedding
            tracked_obj.name = name
            tracked_obj.distances = distances
            tracked_obj.last_recognition = time.time()

    def stop(self):
        self.stop_threads = True
