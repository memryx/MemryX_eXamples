import queue
import numpy as np
import cv2
from PySide6.QtCore import QTimer, QObject, Signal, Qt
from PySide6.QtWidgets import QCheckBox, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
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
        self.paused = False

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
        self.logo_checkbox = QCheckBox("Show Logo")
        self.logo_checkbox.setChecked(True)
        self.logo_scale_slider = SliderWithLabel("Logo Scale:", 
                                                 minimum=1,    # corresponds to 0.01
                                                 maximum=25,   # corresponds to 0.25 (1/4 of screen)
                                                 initial=10,   # default 0.10
                                                 step=1,
                                                 multiplier=0.01)

        # Load icon
        self.load_icons()

    def set_paused(self, paused: bool):
        self.paused = paused

    def load_icons(self):
        icon = cv2.imread("../assets/mx-logo.png", cv2.IMREAD_UNCHANGED)
        if icon is None:
            self.original_logo = None
            self.logo = None
            return
        converted = cv2.cvtColor(icon, cv2.COLOR_RGBA2BGRA) if icon.shape[2] == 4 else icon
        self.original_logo = converted  # keep full-resolution source
        self.logo = converted.copy()

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
        if not getattr(self, "logo_checkbox", None) or not self.logo_checkbox.isChecked():
            return frame  # logo disabled

        if not hasattr(self, "original_logo") or self.original_logo is None:
            return frame  # no logo loaded

        frame_h, frame_w = frame.shape[:2]

        # Determine target width based on slider scale (fraction of frame width)
        scale = self.logo_scale_slider.value()  # expected between 0.01 and 0.25
        target_w = int(frame_w * scale)
        if target_w <= 0:
            return frame

        # Preserve aspect ratio
        orig_h, orig_w = self.original_logo.shape[:2]
        aspect = orig_h / orig_w
        target_h = int(target_w * aspect)

        # Resize logo each frame (use original to avoid quality erosion)
        try:
            resized_logo = cv2.resize(self.original_logo, (target_w, target_h),
                                      interpolation=cv2.INTER_AREA if target_w < orig_w else cv2.INTER_LINEAR)
        except Exception:
            resized_logo = self.original_logo.copy()
            target_h, target_w = orig_h, orig_w

        x, y = frame_w - target_w - 10, 10  # top-right with margin

        # Blend with alpha if present
        if resized_logo.shape[2] == 4:
            alpha_s = resized_logo[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(3):
                frame[y:y+target_h, x:x+target_w, c] = (
                    alpha_s * resized_logo[:, :, c] +
                    alpha_l * frame[y:y+target_h, x:x+target_w, c]
                )
        else:
            frame[y:y+target_h, x:x+target_w] = resized_logo

        return frame

    def draw(self, frame):
        self.framerate.update()
        frame = np.copy(frame)
        frame = self.overlay_icon(frame)
        tracked_objects = self.face_tracker.get_activated_tracker_objects()
        frame = self.draw_objects(frame, tracked_objects)

        # If paused, dim and overlay “Paused: Click to Play”
        if self.paused:
            h, w = frame.shape[:2]
            # Dim the frame
            darkened = np.zeros_like(frame)
            frame = cv2.addWeighted(frame, 0.4, darkened, 0.6, 0)

            # Prepare text
            text = "Paused: Click to Play"
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Scale text relative to width
            font_scale = 1.5
            thickness = 3
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            text_x = (w - text_width) // 2
            text_y = (h + text_height) // 2

            # Draw outline for readability
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, MxColors.LightBlue, thickness, lineType=cv2.LINE_AA)

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

        # Logo config: label, then checkbox + slider horizontally
        layout.addWidget(QLabel("Logo:"))
        logo_layout = QHBoxLayout()
        logo_layout.addWidget(self.compositor.logo_checkbox)
        logo_layout.addWidget(self.compositor.logo_scale_slider)
        layout.addLayout(logo_layout)

        # Buttons for resetting to default values and closing the popup.
        button_layout = QHBoxLayout()
        self.reset_button = QPushButton("Reset to Default")
        self.close_button = QPushButton("Close")
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)

        self.reset_button.clicked.connect(self.reset_defaults)
        self.close_button.clicked.connect(self.close)

        self.compositor.logo_checkbox.toggled.connect(self.on_logo_checkbox_toggled)
        self.on_logo_checkbox_toggled(self.compositor.logo_checkbox.isChecked())

    # handler:
    def on_logo_checkbox_toggled(self, enabled: bool):
        self.compositor.logo_scale_slider.setEnabled(enabled)
        # defensive: if SliderWithLabel wraps inner widgets, enable those too
        inner_slider = getattr(self.compositor.logo_scale_slider, "slider", None)
        if inner_slider is not None:
            inner_slider.setEnabled(enabled)
        inner_label = getattr(self.compositor.logo_scale_slider, "label", None)
        if inner_label is not None:
            inner_label.setEnabled(enabled)

    def reset_defaults(self):
        # Reset checkboxes.
        self.compositor.bbox_checkbox.setChecked(False)
        self.compositor.keypoints_checkbox.setChecked(False)
        self.compositor.distance_checkbox.setChecked(False)

        # Reset other sliders to their initial/default values.
        self.compositor.label_scale_slider.setValue(1.35)
        self.compositor.label_thickness_slider.setValue(4)
        self.compositor.line_thickness_slider.setValue(4)

        # Reset logo
        self.compositor.logo_checkbox.setChecked(True)
        self.on_logo_checkbox_toggled(True)
        self.compositor.logo_scale_slider.setValue(10)  # corresponds to 0.10 because multiplier=0.01


