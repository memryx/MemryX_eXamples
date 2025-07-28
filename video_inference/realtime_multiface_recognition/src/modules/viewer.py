import cv2
from PySide6.QtWidgets import (QLabel, QWidget, QVBoxLayout)
from PySide6.QtGui import QImage, QPixmap, QMouseEvent
from PySide6.QtCore import Signal

# Viewer for video frames  
class FrameViewer(QWidget):
    mouse_move = Signal(tuple)
    mouse_click = Signal(tuple)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Viewer")

        # Create and configure the video display label
        self.video_label = VideoLabel(self)
        self.video_label.setMouseTracking(True)
        #self.video_label.mouseMoveEvent = self.handle_mouse_move

        # Set a layout and add the video label to the widget
        layout = QVBoxLayout(self)
        layout.addWidget(self.video_label)

        self.current_frame = None

    def update_frame(self, frame):
        self.current_frame = frame

        # Resize the frame to fit the available area for the video viewer while preserving the aspect ratio
        video_label_width = self.video_label.width()
        video_label_height = self.video_label.height()
        frame_height, frame_width, _ = frame.shape

        aspect_ratio = frame_width / frame_height
        if video_label_width / video_label_height > aspect_ratio:
            #new_height = min(video_label_height, frame_height)
            new_height = video_label_height
            new_width = int(aspect_ratio * new_height)
        else:
            #new_width = min(video_label_width, frame_width)
            new_width = video_label_width
            new_height = int(new_width / aspect_ratio)

        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        self.video_label.setMinimumSize(1, 1)

        ## Get image information
        height, width, channels = frame.shape
        bytes_per_line = channels * width

        # Create QImage and display it
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

class VideoLabel(QLabel):
    def __init__(self, parent: FrameViewer):
        super().__init__(parent)
        self.setMouseTracking(True)
    
    def mousePressEvent(self, ev: QMouseEvent):
        viewer = self.parent()
        if viewer.current_frame is not None:
            pos = self._translate_mouse_coords(ev, viewer)
            viewer.mouse_click.emit(pos)
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QMouseEvent):
        viewer = self.parent()
        if viewer.current_frame is not None:
            pos = self._translate_mouse_coords(ev, viewer)
            viewer.mouse_move.emit(pos)

        super().mouseMoveEvent(ev)

    def _translate_mouse_coords(self, ev, viewer):
        # Get the position of the mouse relative to the QLabel
        mouse_x = ev.position().x()
        mouse_y = ev.position().y()

        # Calculate the scaling factors
        video_label_width = self.width()
        video_label_height = self.height()
        frame_height, frame_width, _ = viewer.current_frame.shape

        aspect_ratio = frame_width / frame_height
        if video_label_width / video_label_height > aspect_ratio:
            new_height = video_label_height
            new_width = int(aspect_ratio * new_height)
        else:
            new_width = video_label_width
            new_height = int(new_width / aspect_ratio)

        x_offset = 0#(video_label_width - new_width) // 2
        y_offset = (video_label_height - new_height) // 2

        # Adjust mouse position to account for the resized video and offset
        if x_offset <= mouse_x <= x_offset + new_width and y_offset <= mouse_y <= y_offset + new_height:
            adjusted_x = int((mouse_x - x_offset) * (frame_width / new_width))
            adjusted_y = int((mouse_y - y_offset) * (frame_height / new_height))
            mouse_position = (adjusted_x, adjusted_y)
        else:
            mouse_position = None

        return mouse_position
