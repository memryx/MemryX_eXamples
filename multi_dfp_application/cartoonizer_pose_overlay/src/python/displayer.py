import time
from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QCheckBox,
    QHBoxLayout,
    QVBoxLayout,
    QSizePolicy,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import cv2
import threading
import os
import numpy as np
from constant import *

# Optional: fix Qt plugin path for your environment
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = (
    "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"
)


class DisplayerWithCheckboxes(QWidget):
    def __init__(self, dfp_names, display_fps, window_title="DFP Overlay Display"):
        super().__init__()

        self.dfp_names = dfp_names
        self.display_fps = display_fps

        # QLabel for displaying video frame
        self.video_label = QLabel()
        self.video_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Checkboxes (aligned to top-left of video)
        self.checkboxes = []
        for dfp_name in dfp_names:
            checkbox = QCheckBox(dfp_name)
            checkbox.stateChanged.connect(self.on_checkbox_state_changed)
            self.checkboxes.append(checkbox)

        # Add a dictionary to store checkbox states
        self.checkbox_states = {dfp_name: False for dfp_name in dfp_names}

        checkbox_layout = QVBoxLayout()
        checkbox_layout.setContentsMargins(10, 10, 10, 10)
        checkbox_layout.setSpacing(10)
        for checkbox in self.checkboxes:
            checkbox_layout.addWidget(checkbox)

        # Wrapper layout for aligning checkboxes to top
        left_wrapper = QWidget()
        left_wrapper.setLayout(checkbox_layout)

        # Main layout: horizontal (checkboxes on left, video on right)
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        main_layout.addWidget(left_wrapper, alignment=Qt.AlignCenter)
        main_layout.addWidget(self.video_label)

        self.setLayout(main_layout)
        self.setWindowTitle(window_title)
        self.setFixedSize(800, 450)

        # Timer for video update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.display_interval_ms = 1000 // display_fps
        self.timer.start(self.display_interval_ms)

        # results includes [raw_frame, cartoonized_frame, pose_estimation]
        self.buffer_lock = threading.Lock()
        self.display_buffer = {}
        self.curr_display_id = 0

        # Lock for checkbox state updates
        self.checkbox_lock = threading.Lock()

    def update_buffer(self, frame_id, dfp_name, result_img):

        with self.buffer_lock:
            # Only update if frame_id is greater than current display id, otherwise discard the result
            if frame_id < self.curr_display_id:
                return

            """
                Format of display_buffer
                
                self.display_buffer = {
                    frame_id1: {dfp_name1: img, dfp_name2: img, ...},
                    frame_id2: {dfp_name1: img, dfp_name2: img, ...},
                    ...
                }
            """
            self.display_buffer.setdefault(frame_id, {})[dfp_name] = result_img

    def update_frame(self):

        display_img = None
        with self.buffer_lock:
            if not self.curr_display_id in self.display_buffer:
                time.sleep(0.05)
                return

            # Check if all DFP results are ready for the current display frame
            curr_display_buffer = self.display_buffer[self.curr_display_id]

            # get raw frame
            raw_frame = curr_display_buffer[RAW_FRAME_NAME]
            display_img = raw_frame.copy()

            # get DFP results according to checked checkboxes
            for key in self.get_checked_DFP_names():
                if key not in curr_display_buffer:
                    time.sleep(0.05)
                    return

                dfp_result = curr_display_buffer[key]

                # Compute absolute difference
                diff = cv2.absdiff(raw_frame, dfp_result)

                # Use the diff mask to copy pixels from dfp_result to display_img
                display_img = np.where(diff > 0, dfp_result, display_img)

        # convert to QImage and display
        rgb_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        q_img = QImage(rgb_img.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        self.video_label.setPixmap(pixmap)
        self.video_label.setFixedSize(pixmap.size())

        # Clear result of frame after displaying
        with self.buffer_lock:
            del self.display_buffer[self.curr_display_id]
            self.curr_display_id += 1

    def on_checkbox_state_changed(self, state):
        checkbox = self.sender()
        label = checkbox.text()
        is_checked = state == Qt.Checked

        with self.checkbox_lock:
            self.checkbox_states[label] = is_checked
            print(f"{label} is {'checked' if is_checked else 'unchecked'}")
            print("Current states:", self.checkbox_states)

    def get_checked_DFP_names(self):
        with self.checkbox_lock:
            return [name for name, checked in self.checkbox_states.items() if checked]

    def closeEvent(self, event):
        event.accept()
