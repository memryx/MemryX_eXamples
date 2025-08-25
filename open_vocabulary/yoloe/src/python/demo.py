import sys
import argparse
from pathlib import Path
import queue
import time
import cv2
import numpy as np

from PySide6.QtWidgets import  (QApplication, QLabel, QMainWindow, QWidget,
                                QVBoxLayout, QHBoxLayout, QPushButton)
from PySide6.QtGui import QImage, QPixmap, QMouseEvent
from PySide6.QtCore import QThread, Signal

from YoloE import YoloE, AnnotatedFrame,Framedata
from gui.controls import ControlPanel
from gui.progress_bar import ProgressBarWidget

# Thread for reading video frames
class VideoReaderThread(QThread):
    def __init__(self, video_source, frame_queue, bypass_frame_queue):
        super().__init__()
        self.video_source = video_source
        self.frame_queue = frame_queue
        self.bypass_frame_queue = bypass_frame_queue
        self.stop_threads = False
        self.cur_frame = None
        self.static_image = np.zeros([1080,1920,3], dtype=np.uint8)
        self.do_inference = True 
        self.do_seg = True
        self.do_det = True
        self.do_label = True
        self.do_blur = False
        self.remove_background = False


    def toggle_inference(self, force=None):
        if force is None:
            self.do_inference = not self.do_inference
        else:
            self.do_inference = force
    
    def toggle_blur(self, force=None):
        if force is None:
            self.do_blur = not self.do_blur
            if self.do_blur:
                self.remove_background = False  # Ensure exclusivity
        else:
            self.do_blur = force
            if force:
                self.remove_background = False  # Ensure exclusivity

    def toggle_remove_background(self, force=None):
        if force is None:
            self.remove_background = not self.remove_background
            if self.remove_background:
                self.do_blur = False  # Ensure exclusivity
        else:
            self.remove_background = force
            if force:
                self.do_blur = False  # Ensure exclusivity

    def toggle_segmentation(self, force=None):
        if force is None:
            self.do_seg = not self.do_seg
        else:
            self.do_seg = force

    def toggle_label(self, force=None):
        if force is None:
            self.do_label = not self.do_label
        else:
            self.do_label = force

    def toggle_detection_box(self, force=None):
        if force is None:
            self.do_det = not self.do_det
        else:
            self.do_det = force

    def run(self):
        # Handle video case
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video source: {self.video_source}. "
                             f"Please check the video path or try a different video source.")

        while not self.stop_threads:
            ret, frame = cap.read()
            if not ret:
                print("Stream ended or failed to grab frame.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.cur_frame = np.array(rgb_frame)

            try:
                if self.do_inference:
                    self.frame_queue.put(Framedata(self.cur_frame,self.do_seg,self.do_det,self.do_label,self.remove_background,self.do_blur), timeout=0.033)
                else:
                    self.bypass_frame_queue.put(self.cur_frame, timeout=0.033)
            except queue.Full:
                pass

        cap.release()
        # print("Video reader stopped.")

    def stop(self):
        self.stop_threads = True

class VideoDisplayThread(QThread):
    frame_ready = Signal(np.ndarray)

    def __init__(self, frame_queue, bypass_frame_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.bypass_frame_queue = bypass_frame_queue
        self.stop_threads = False
        self.do_inference = True 

    def toggle_inference(self, force=None):
        if force is None:
            self.do_inference = not self.do_inference
        else:
            self.do_inference = force

    def run(self):
        while not self.stop_threads or not self.frame_queue.empty():
            try:
                if self.do_inference:
                    annotated_frame = self.frame_queue.get(timeout=0.1)  # Timeout to allow shutdown
                else:
                    frame = self.bypass_frame_queue.get(timeout=0.1)  # Timeout to allow shutdown
                    annotated_frame = AnnotatedFrame(frame)
                self.frame_ready.emit(annotated_frame)
            except queue.Empty:
                continue

        # print("Video display stopped.")

    def stop(self):
        self.stop_threads = True

class UpdateClasses(QThread):
    def __init__(self):
        super().__init__()
        self.classes = ['person']

    def set_classes(self, classes):
        self.classes = classes

    def run(self):
        mxyoloe.update_classes(self.classes)
        print(f"Updated classes: {self.classes}")

class VideoPlayer(QMainWindow):
    def __init__(self, video_path='/dev/video0'):
        super().__init__()
        self.video_path = video_path
        self.setWindowTitle("Video Player Loop")
        self.resize(1200, 800)

        # Apply dark mode stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
                color: #FFFFFF;
            }
            QLabel {
                color: #FFFFFF;
            }
            QPushButton {
                background-color: #1E1E1E;
                color: #FFFFFF;
                border: 1px solid #333333;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #333333;
            }
            QPushButton:pressed {
                background-color: #555555;
            }
            QWidget {
                background-color: #121212;
                color: #FFFFFF;
            }
            QSlider::groove:horizontal {
                background: #333333;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #555555;
                border: 1px solid #777777;
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #777777;
            }
        """)

        # Set up main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.central_widget.setMinimumSize(300, 200)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Video viewer widget
        self.video_widget = QWidget()
        self.video_layout = QVBoxLayout(self.video_widget)
        self.main_layout.addWidget(self.video_widget)

        # Video display label
        self.video_label = QLabel(self)
        self.video_layout.addWidget(self.video_label)

        # Progress Bar
        self.progress_bar = ProgressBarWidget(n_seconds=30)
        self.progress_bar.setVisible(False)
        self.video_layout.addWidget(self.progress_bar)

        # Control Panel
        self.control_panel = ControlPanel(mxyoloe._conf, mxyoloe._iou, mxyoloe.classes)
        self.control_panel.setMaximumWidth(200)
        self.main_layout.addWidget(self.control_panel)

        # Connect control panel controls
        self.control_panel.class_entry_widget.apply_button.clicked.connect(self.update_classes)
        self.control_panel.class_entry_widget.toggled.connect(self.set_selected_classes)
    
        self.control_panel.buttons.remove_background_button.clicked.connect(self.toggle_remove_background)
        self.control_panel.buttons.do_blur_button.clicked.connect(self.toggle_do_blur)
        self.control_panel.buttons.segmentation_button.clicked.connect(self.toggle_segmentation)
        self.control_panel.buttons.detectionbox_button.clicked.connect(self.toggle_detection_box)
        self.control_panel.buttons.label_button.clicked.connect(self.toggle_label)
        self.control_panel.buttons.value_changed.connect(self.set_visual_config)

        # Set up video-related attributes
        self.frame_queue = mxyoloe
        self.bypass_frame_queue = queue.Queue(maxsize=2)

        self.video_reader_thread = VideoReaderThread(self.video_path, self.frame_queue, self.bypass_frame_queue)
        self.video_display_thread = VideoDisplayThread(self.frame_queue, self.bypass_frame_queue)

        # Connect signals to slots
        self.video_display_thread.frame_ready.connect(self.update_frame)

        # Start threads
        self.video_reader_thread.start()
        self.video_display_thread.start()

        # Class Updater
        self.update_thread = UpdateClasses()
        self.update_thread.finished.connect(self.update_classes_finished)

        self.current_frame = None
        self.annotated_frame = None
        self.timestamps = [0.0] * 30

    def toggle_do_blur(self):
        self.video_reader_thread.toggle_blur()

    def toggle_remove_background(self):
        self.video_reader_thread.toggle_remove_background()

    def toggle_segmentation(self):
        self.video_reader_thread.toggle_segmentation()

    def toggle_label(self):
        self.video_reader_thread.toggle_label()

    def toggle_detection_box(self):
        self.video_reader_thread.toggle_detection_box()
        

    def toggle_inference(self):
        self.video_reader_thread.toggle_inference()
        self.video_display_thread.toggle_inference()


    def set_selected_classes(self, selected_classes):
        mxyoloe.selected_classes = selected_classes

    def set_visual_config(self, config_dict):
        mxyoloe._darkness = config_dict.get('darkness', 0.5)  
        mxyoloe._blur_sigma = config_dict.get('blur_sigma', 10)  

        

    def update_classes(self):
        """
        Changes the underlying model classes
        """
        # Stop inferencing and Streaming / Disable the buttons

        if self.control_panel.buttons.segmentation_button.isEnabled():
            self.control_panel.buttons.segmentation_button.click()

        if self.control_panel.buttons.detectionbox_button.isEnabled():
            self.control_panel.buttons.detectionbox_button.click()

        if self.control_panel.buttons.label_button.isEnabled():
            self.control_panel.buttons.label_button.click()

        self.control_panel.buttons.segmentation_button.setEnabled(False)
        self.control_panel.buttons.detectionbox_button.setEnabled(False)
        self.control_panel.buttons.label_button.setEnabled(False)

        # Swap video_label with progress_bar and start progress bar
        self.video_label.setVisible(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.start()

        # Collect new classes and update
        classes = self.control_panel.class_entry_widget.get_classes()
        self.control_panel.class_entry_widget.disable_apply()
        self.update_thread.set_classes(classes) 
        self.update_thread.start()

    def update_classes_finished(self):
        # Re-Enable the streams
        self.control_panel.buttons.segmentation_button.setEnabled(True)
        self.control_panel.buttons.detectionbox_button.setEnabled(True)
        self.control_panel.buttons.label_button.setEnabled(True)

        self.control_panel.buttons.segmentation_button.click()
        self.control_panel.buttons.detectionbox_button.click()
        self.control_panel.buttons.label_button.click()

        self.control_panel.class_entry_widget.enable_apply()

        # Swap progress_bar back with video_label
        self.progress_bar.setVisible(False)
        self.progress_bar.reset()
        self.video_label.setVisible(True)

        # Reset the selected classes
        self.control_panel.class_entry_widget._on_entry_toggled()

    def update_frame(self, annotated_frame):
        cur_time = time.time()
        self.timestamps.append(cur_time)
        self.timestamps.pop(0)
        dt = np.average([self.timestamps[i + 1] - self.timestamps[i] for i in range(len(self.timestamps) - 1)])

        self.current_frame = annotated_frame.image
        self.annotated_frame = annotated_frame

        # Draw bounding boxes and labels for each face in the frame
        frame = annotated_frame.image#.copy()
        if not isinstance(frame, np.ndarray):
            return

        # Resize the frame to fit the available area for the video viewer while preserving the aspect ratio
        video_label_width = self.video_label.width()
        video_label_height = self.video_label.height()
        frame_height, frame_width, _ = frame.shape

        aspect_ratio = frame_width / frame_height
        if video_label_width / video_label_height > aspect_ratio:
            new_height = video_label_height
            new_width = int(aspect_ratio * new_height)
        else:
            new_width = video_label_width
            new_height = int(new_width / aspect_ratio)

        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        self.video_label.setMinimumSize(1, 1)

        # Get image information
        height, width, channels = frame.shape
        bytes_per_line = channels * width

        # Create QImage and display it
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))


    def closeEvent(self, event):
        # Stop threads before closing
        self.video_reader_thread.stop()
        self.video_reader_thread.wait()
        self.video_display_thread.stop()
        self.video_display_thread.wait()
        mxyoloe.stop()
        event.accept()

def parse_args():
    parser = argparse.ArgumentParser(description="Compile script for YoloEDemo")
    
    parser.add_argument(
        "-i", "--input_video_path",
        type=str,
        default="/dev/video0",
        help="Path to the video file or camera device (default: /dev/video0)."
    )

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    app = QApplication(sys.argv)
    global mxyoloe 
    mxyoloe = YoloE() 

    player = VideoPlayer(args.input_video_path)
    player.show()
    sys.exit(app.exec())

