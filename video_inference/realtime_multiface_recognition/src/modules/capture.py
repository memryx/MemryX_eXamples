import sys
import queue
import cv2
import numpy as np

from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QWidget,
                               QVBoxLayout, QLineEdit, QPushButton,
                               QHBoxLayout, QSplitter, QCheckBox, QFrame,
                               QTreeWidget, QTreeWidgetItem, QInputDialog,
                               QMessageBox, QFileDialog, QDialog, QComboBox, QFormLayout)
from PySide6.QtGui import QImage, QPixmap, QMouseEvent, QKeyEvent
from PySide6.QtCore import QTimer, Qt, QThread, Signal, QMutex


import time

from .utils import Framerate

def is_image(source):
    return source.endswith(('.jpg', '.jpeg', '.png', '.bmp'))

def is_video(source):
    return source.endswith(('.mp4', '.webm', 'mkv'))

class VideoConfig:
    def __init__(self, width=3840, height=2160, fourcc='MJPG', fps=30):
        self.config = {
            cv2.CAP_PROP_FRAME_WIDTH: width,
            cv2.CAP_PROP_FRAME_HEIGHT: height,
            cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*fourcc),
            cv2.CAP_PROP_FPS: fps,
        }

    def set(self, cap: cv2.VideoCapture):
        for k, v in self.config.items():
            cap.set(k, v)

VIDEO_CONFIG = {
    '4k': VideoConfig(3840, 2160), 
    '2k': VideoConfig(2560, 1440),
    '1080p': VideoConfig(1920,1080, fps=60),
    '720p': VideoConfig(1280, 720)
}

# Thread for reading video frames
class CaptureThread(QThread):
    frame_ready = Signal(np.ndarray)

    def __init__(self, video_source, video_config=None):
        super().__init__()
        self.video_config = video_config
        self.video_source = video_source
        self.stop_threads = False
        self.pause = False
        self.cur_frame = None

        self.framerate = Framerate()

    def _read_image(self):
        """Read image file and simulate video stream"""
        frame = cv2.imread(self.video_source)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frame is None:
            print("Failed to load image.")
            return

        while not self.stop_threads:
            self.framerate.update()
            self.frame_ready.emit(frame)
            time.sleep(1 / 120)

    def _read_stream(self):
        """Read video file or stream"""

        # Handle video case
        cap = cv2.VideoCapture(self.video_source)
        if self.video_config is not None: 
            self.video_config.set(cap)

        while not self.stop_threads:
            self.framerate.update()
            if self.pause:
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                print("Stream ended or failed to grab frame.")
                break

            frame = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.frame_ready.emit(frame)

        cap.release()

    def _read_video(self):
        """Read stream and throttle to the video’s native FPS."""
        cap = cv2.VideoCapture(self.video_source)
        # grab the file’s native FPS; fall back to 30 if we can’t read it
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = 1.0 / fps

        last_time = time.time()
        while not self.stop_threads:
            # if paused, just spin-sleep
            if self.pause:
                time.sleep(0.1)
                continue

            start = time.time()
            ret, frame = cap.read()
            if not ret:
                # loop back to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # convert & emit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_ready.emit(np.array(frame))

            # throttle: sleep the remainder of frame_interval
            elapsed = time.time() - start
            to_wait = frame_interval - elapsed
            if to_wait > 0:
                time.sleep(to_wait)

            last_time = start

        cap.release()


    def run(self):
        """Read video frames and emit signal"""
        if is_image(self.video_source):
            self._read_image()
        elif is_video(self.video_source):
            self._read_video()
        else:
            self._read_stream()

    def toggle_play(self):
        self.pause = not self.pause

    def stop(self):
        print("Shutting down CaptureThread")
        self.stop_threads = True

# New dialog for capture thread configuration
class CaptureConfigDialog(QDialog):
    def __init__(self, current_video_path, current_resolution, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Capture Thread Configuration")
        self.setup_ui(current_video_path, current_resolution)

    def setup_ui(self, current_video_path, current_resolution):
        layout = QVBoxLayout(self)

        form_layout = QFormLayout()
        self.video_path_edit = QLineEdit(self)
        self.video_path_edit.setText(current_video_path)
        form_layout.addRow("Video Path:", self.video_path_edit)

        self.resolution_combo = QComboBox(self)
        self.resolution_combo.addItems(["1080p", "2k", "4k"])
        index = self.resolution_combo.findText(current_resolution)
        if index != -1:
            self.resolution_combo.setCurrentIndex(index)
        form_layout.addRow("Resolution:", self.resolution_combo)

        layout.addLayout(form_layout)

        # Buttons for Cancel and Apply
        button_layout = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel", self)
        self.apply_button = QPushButton("Apply", self)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.apply_button)
        layout.addLayout(button_layout)

        self.cancel_button.clicked.connect(self.reject)
        self.apply_button.clicked.connect(self.accept)

    def get_configuration(self):
        return self.video_path_edit.text(), self.resolution_combo.currentText()

# Existing configuration panel (using compositor checkboxes)


