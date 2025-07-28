import queue
import numpy as np
import cv2
from PySide6.QtCore import QTimer, QObject, Signal, Qt
from PySide6.QtWidgets import QCheckBox, QDialog, QVBoxLayout, QHBoxLayout, QPushButton
from .utils import Framerate, SliderWithLabel

class MxColors: 
    LightBlue = (198, 234, 242)
    Blue = (53, 169, 188)
    DarkBlue = (20, 53, 85) 
    Teal = (60, 187, 187)

class Compositor(QObject):
    frame_ready = Signal(np.ndarray)

    def __init__(self, face_tracker, parent=None):
        super().__init__(parent)
        self.face_tracker = face_tracker
        self.framerate = Framerate()
        self.mouse_position = (-1, -1)

        # Create config widgets.
        self.bbox_checkbox = QCheckBox("Draw Boxes")
        self.bbox_checkbox.setChecked(True)
        self.keypoints_checkbox = QCheckBox("Draw Keypoints")
        self.keypoints_checkbox.setChecked(False)
        self.distance_checkbox = QCheckBox("Show Similarity")
        self.distance_checkbox.setChecked(False)

        self.label_scale_slider = SliderWithLabel("Font Scale:", 
                                                  minimum=50, 
                                                  maximum=300, 
                                                  initial=85, 
                                                  step=5, 
                                                  multiplier=0.01)
        self.label_thickness_slider = SliderWithLabel("Font Thickness:", 
                                                      minimum=1,
                                                      maximum=10, 
                                                      initial=2,
                                                      step=1, 
                                                      multiplier=1)
        self.line_thickness_slider = SliderWithLabel("Line Thickness:", 
                                                     minimum=1,
                                                     maximum=10, 
                                                     initial=2,
                                                     step=1, 
                                                     multiplier=1)

        # Load icon
        self.load_icons()

    def load_icons(self):
        icon = cv2.imread("../assets/mx-logo.png", cv2.IMREAD_UNCHANGED)
        #icon = cv2.resize(icon, (100, 100))
        self.logo = cv2.cvtColor(icon, cv2.COLOR_RGBA2BGRA)

    def update_mouse_pos(self, pos):
        self.mouse_position = pos

    def draw_name_plain(self, frame, obj):
        (left, top, right, bottom) = obj.bbox
        label = f'{obj.name}({obj.track_id})'
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85, MxColors.Blue, 2)

    def draw_name(self, frame, obj):
        (left, top, right, bottom) = obj.bbox
        label = f'{obj.name}'#({obj.track_id})' # TODO add trackid check box
        h, w = frame.shape[:2]

        # Get dynamic font parameters from slider widgets.
        font_scale = self.label_scale_slider.value()
        thickness = int(self.label_thickness_slider.value())

        cx = left + (right - left) // 2
        diag_length = (right - left) // 2
        margin = 10
        line_thickness = int(self.line_thickness_slider.value())

        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        horiz_length = text_width + margin

        dx = 1
        upward = True
        start = (cx, top)
        diag_end = (cx + dx * diag_length, top - diag_length)
        upward_text_y = diag_end[1] - 10

        if upward_text_y < 0:
            upward = False

        if upward:
            start = (cx, top)
            diag_end = (cx + dx * diag_length, top - diag_length)
            text_y = diag_end[1] - 10
        else:
            start = (cx, bottom)
            diag_end = (cx + dx * diag_length, bottom + diag_length)
            text_y = diag_end[1] + text_height + 10

        if dx == 1 and (diag_end[0] + horiz_length > w):
            dx = -1
        elif dx == -1 and (diag_end[0] - horiz_length < 0):
            dx = 1

        if upward:
            diag_end = (cx + dx * diag_length, top - diag_length)
        else:
            diag_end = (cx + dx * diag_length, bottom + diag_length)

        horiz_end = (int(diag_end[0] + dx * horiz_length), diag_end[1])
        center_horiz = diag_end[0] + dx * (horiz_length / 2)
        text_x = int(center_horiz - text_width / 2)

        cv2.line(frame, (cx, top) if upward else (cx, bottom), 
                 (int(diag_end[0]), int(diag_end[1])), MxColors.DarkBlue, thickness=line_thickness)
        cv2.line(frame, (int(diag_end[0]), int(diag_end[1])), 
                 (int(horiz_end[0]), int(horiz_end[1])), MxColors.DarkBlue, thickness=line_thickness)

        cv2.putText(frame, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, MxColors.Blue, thickness)

    def draw_objects(self, frame, tracked_objects):
        for obj in tracked_objects:
            (left, top, right, bottom) = obj.bbox

            self.draw_name(frame, obj)
            if self.distance_checkbox.isChecked():
                for i, (name, distance) in enumerate(obj.distances):
                    if i == 3:
                        break
                    label = f'{name}: {distance:.1f}'
                    cv2.putText(frame, label, (left + 10, top + 10 + 20 * i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

            if self.bbox_checkbox.isChecked(): 
                line_thickness = int(self.line_thickness_slider.value())
                cv2.rectangle(frame, (left, top), (right, bottom), MxColors.Blue, line_thickness)

            if self.keypoints_checkbox.isChecked():
                for (x, y) in obj.keypoints:
                    cv2.circle(frame, (x, y), 5, MxColors.LightBlue, -1)

            if self.mouse_position:
                mouse_x, mouse_y = self.mouse_position
                if left <= mouse_x <= right and top <= mouse_y <= bottom:
                    overlay = frame.copy()
                    alpha = 0.5
                    cv2.rectangle(overlay, (left, top), (right, bottom), MxColors.Blue, -1)
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def overlay_icon(self, frame):
        x, y = frame.shape[1] - self.logo.shape[1] - 10, 10
        h, w = self.logo.shape[:2]
        if self.logo.shape[2] == 4:
            alpha_s = self.logo[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(3):
                frame[y:y+h, x:x+w, c] = (
                    alpha_s * self.logo[:, :, c] +
                    alpha_l * frame[y:y+h, x:x+w, c]
                )
        else:
            frame[y:y+h, x:x+w] = self.logo
        return frame

    def draw(self, frame):
        self.framerate.update()
        frame = np.copy(frame)
        frame = self.overlay_icon(frame)
        tracked_objects = self.face_tracker.get_activated_tracker_objects()
        frame = self.draw_objects(frame, tracked_objects)
        self.frame_ready.emit(frame)

class CompositorConfigPopup(QDialog):
    def __init__(self, compositor, parent=None):
        super().__init__(parent)
        self.compositor = compositor
        self.setWindowTitle("Compositor Configuration")
        self.setup_ui()
        self.reset_defaults()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Add (and reparent) the compositor's configuration widgets.
        layout.addWidget(self.compositor.bbox_checkbox)
        layout.addWidget(self.compositor.keypoints_checkbox)
        layout.addWidget(self.compositor.distance_checkbox)
        layout.addWidget(self.compositor.label_scale_slider)
        layout.addWidget(self.compositor.label_thickness_slider)
        layout.addWidget(self.compositor.line_thickness_slider)

        # Buttons for resetting to default values and closing the popup.
        button_layout = QHBoxLayout()
        self.reset_button = QPushButton("Reset to Default")
        self.close_button = QPushButton("Close")
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)

        self.reset_button.clicked.connect(self.reset_defaults)
        self.close_button.clicked.connect(self.close)
    
    def reset_defaults(self):
        # Reset checkboxes.
        self.compositor.bbox_checkbox.setChecked(False)
        self.compositor.keypoints_checkbox.setChecked(False)
        self.compositor.distance_checkbox.setChecked(False)
        # Reset sliders to their initial/default values.
        self.compositor.label_scale_slider.setValue(1.35)
        self.compositor.label_thickness_slider.setValue(4)
        self.compositor.line_thickness_slider.setValue(4)

