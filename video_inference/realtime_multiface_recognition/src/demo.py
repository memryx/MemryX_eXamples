import os
import sys
import glob
import cv2
import numpy as np

from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QWidget,
                               QVBoxLayout, QLineEdit, QPushButton,
                               QHBoxLayout, QSplitter, QCheckBox, QFrame,
                               QTreeWidget, QTreeWidgetItem, QInputDialog, QDialog,
                               QMessageBox, QFileDialog)
from PySide6.QtGui import QImage, QPixmap, QMouseEvent, QKeyEvent
from PySide6.QtCore import QTimer, Qt, QThread, Signal, QMutex

import time
from pathlib import Path

from modules.capture import CaptureThread, CaptureConfigDialog, VIDEO_CONFIG
from modules.compositor import Compositor, CompositorConfigPopup
from modules.viewer import FrameViewer
from modules.database import FaceDatabase, DatabaseViewerWidget
from modules.tracker import FaceTracker

class Demo(QMainWindow):
    def __init__(self, video_path='/dev/video0', video_config=None):
        super().__init__()
        self.setWindowTitle("Video Viewer")
        
        # Create the video display.
        self.viewer = FrameViewer()

        # Set up video capture and processing.
        self.capture_thread = CaptureThread(video_path, video_config)
        self.face_database = FaceDatabase()
        self.database_viewer = DatabaseViewerWidget(self.face_database)
        self.tracker = FaceTracker(self.face_database)
        self.compositor = Compositor(self.tracker)
        self.compositor.set_paused(self.capture_thread.pause)

        # Create a button to open the compositor config popup.
        self.config_popup_button = QPushButton("Compositor Config", self)
        self.config_popup_button.clicked.connect(self.open_compositor_config)

        self.capture_control_button = QPushButton("Capture Config", self)
        self.capture_control_button.clicked.connect(self.open_capture_config)

        # Wire signals.
        self.capture_thread.frame_ready.connect(self.tracker.detect)
        self.tracker.frame_ready.connect(self.compositor.draw)
        self.compositor.frame_ready.connect(self.viewer.update_frame)
        self.viewer.mouse_move.connect(self.compositor.update_mouse_pos)
        self.viewer.mouse_click.connect(self.handle_viewer_mouse_click)

        # Add click-to-pause functionality: clicking anywhere in the viewer toggles capture play/pause
        self.viewer.mouse_click.connect(self.toggle_capture_pause)

        self.setup_layout()

        self.tracker.start()
        self.capture_thread.start()
        self.timestamps = [0] * 30

        self.fps_timer = QTimer(self)
        self.fps_timer.setInterval(500)
        self.fps_timer.timeout.connect(self.poll_framerates)
        self.fps_timer.start()

        # Create a persistent instance for the config popup.
        self.config_popup = None

    def setup_layout(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setMinimumSize(300, 200)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Splitter to separate control panel and video viewer.
        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)

        # Left control panel.
        self.control_panel = QWidget()
        self.control_panel.setFixedWidth(300)
        self.control_layout = QVBoxLayout(self.control_panel)
        self.splitter.addWidget(self.control_panel)

        # Add logo at the top of control panel
        logo_label = QLabel()
        logo_pixmap = QPixmap("../assets/mx-logo.png")
        logo_label.setPixmap(logo_pixmap.scaledToWidth(200, Qt.SmoothTransformation))
        logo_label.setAlignment(Qt.AlignCenter)
        self.control_layout.addWidget(logo_label)

        # Add the capture config and compositor config buttons.
        self.control_layout.addWidget(self.capture_control_button)
        self.control_layout.addWidget(self.config_popup_button)
        self.control_layout.addWidget(self.database_viewer)

        # Right: Video viewer.
        self.splitter.addWidget(self.viewer)
        self.splitter.setStretchFactor(1, 1)

    def open_compositor_config(self):
        if self.config_popup is None:
            self.config_popup = CompositorConfigPopup(self.compositor, self)
        self.config_popup.show()
        self.config_popup.raise_()

    def open_capture_config(self):
        current_resolution = "2k"
        for res in ["1080p", "2k", "4k"]:
            if VIDEO_CONFIG.get(res) == self.capture_thread.video_config:
                current_resolution = res
                break

        dialog = CaptureConfigDialog(self.capture_thread.video_source, current_resolution, self)
        if dialog.exec() == QDialog.Accepted:
            new_video_path, new_resolution = dialog.get_configuration()
            print(f"Applying new capture configuration: {new_video_path}, {new_resolution}")
            self.capture_thread.stop()
            self.capture_thread.wait()
            new_config = VIDEO_CONFIG.get(new_resolution, self.capture_thread.video_config)
            self.capture_thread = CaptureThread(new_video_path, new_config)
            self.capture_thread.frame_ready.connect(self.tracker.detect)
            self.capture_thread.start()

    def handle_viewer_mouse_click(self, mouse_pos):
        # Existing face-capture logic
        if mouse_pos is None:
            return
        
        tracker_frame = np.copy(self.tracker.current_frame.image)
        tracker_objects = self.tracker.get_activated_tracker_objects()

        found = False 
        mouse_x, mouse_y = mouse_pos
        for obj in tracker_objects:
            (left, top, right, bottom) = obj.bbox
            if left <= mouse_x <= right and top <= mouse_y <= bottom:
                width = right - left
                height = bottom - top
                margin = 10
                bbox_size = max(width, height) + 2 * margin
                center_x, center_y = left + width // 2, top + height // 2
                x_start = max(0, center_x - bbox_size // 2)
                x_end = min(tracker_frame.shape[1], center_x + bbox_size // 2)
                y_start = max(0, center_y - bbox_size // 2)
                y_end = min(tracker_frame.shape[0], center_y + bbox_size // 2)
                cropped_frame = tracker_frame[y_start:y_end, x_start:x_end]

                profile_path = self.database_viewer.get_selected_directory()
                if not profile_path:
                    if obj.name == 'Unknown':
                        new_profile = self.database_viewer.add_profile()
                        profile_path = os.path.join(self.database_viewer.db_path, new_profile)
                    else:
                        profile_path = os.path.join(self.database_viewer.db_path, obj.name)

                if os.path.exists(profile_path) and Path(profile_path) != Path(self.database_viewer.db_path):
                    i = 0
                    while os.path.exists(os.path.join(profile_path, f"{i}.jpg")):
                        i += 1
                    filename = os.path.join(profile_path, f"{i}.jpg")
                    print(f'Saving image to {filename}')
                    cv2.imwrite(filename, cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR))
                    self.database_viewer.load_profiles()
                    self.face_database.add_to_database(obj.embedding, filename)
                found = True
                break

    def toggle_capture_pause(self, pos):
            """Toggle play/pause of the capture thread when the viewer is clicked, 
            but only if the click is NOT over a face."""
            if pos is None:
                return

            mouse_x, mouse_y = pos
            # If click is on any active tracked face, do not toggle pause.
            for obj in self.tracker.get_activated_tracker_objects():
                left, top, right, bottom = obj.bbox
                if left <= mouse_x <= right and top <= mouse_y <= bottom:
                    return  # clicked on a face; preserve current paused state

            # Click was not on a face: toggle pause/play.
            self.capture_thread.toggle_play()
            state = "paused" if self.capture_thread.pause else "running"
            print(f"Capture thread {state}")
            self.compositor.set_paused(self.capture_thread.pause)

    def poll_framerates(self):
        # If paused, redraw the last known frame so the overlay shows.
        if self.capture_thread.pause:
            if hasattr(self.tracker, "current_frame") and self.tracker.current_frame is not None:
                frame = np.copy(self.tracker.current_frame.image)
                self.compositor.draw(frame)

    def closeEvent(self, event):
        self.capture_thread.stop()
        self.capture_thread.wait()
        self.tracker.stop()
        self.fps_timer.stop()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    video_path = "/dev/video0"  # Update this path as needed.
    streams = sorted(glob.glob('/dev/video*'))

    if not streams:
        video_path = '../assets/mx-logo.png'
    else:
        video_path = streams[0] 

    player = Demo(video_path, VIDEO_CONFIG['2k'])
    player.resize(1200, 800)
    player.show()
    sys.exit(app.exec())

