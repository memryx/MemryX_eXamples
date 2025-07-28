import sys
import cv2
import numpy as np
import os

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QImage, QPainter


# Optional: fix Qt plugin path for your environment
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"


class Displayer(QWidget):
    def __init__(self, num_windows=2):
        super().__init__()
        self.setWindowTitle("Two Video Streams with PyQt5")
        self.left_frame = None
        self.right_frame = None
        self.num_windows = num_windows
        self.resize(1280, 480)  # Adjusted height to match content only

    def update_left(self, frame):
        self.left_frame = frame
        self.update()  # Triggers repaint

    def update_right(self, frame):
        self.right_frame = frame
        self.update()

    def paintEvent(self, event):
        """Paints video frames onto the widget."""
        painter = QPainter(self)
        width, height = 640, 480

        if self.left_frame is not None:
            left_image = self.convert_frame(self.left_frame)
            painter.drawImage(QRect(0, 0, width, height), left_image)

        if self.right_frame is not None:
            right_image = self.convert_frame(self.right_frame)
            painter.drawImage(QRect(width, 0, width, height), right_image)

    def convert_frame(self, frame):
        """Converts OpenCV BGR image to QImage."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
