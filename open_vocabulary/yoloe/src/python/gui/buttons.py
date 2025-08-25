from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QSlider, QLineEdit, QHBoxLayout
)
from PySide6.QtCore import Qt, Signal
import sys


class ToggleButtonWidget(QWidget):
    value_changed = Signal(dict)  # Emits the full control state

    def __init__(self, parent=None):
        super().__init__(parent)
        vbox = QVBoxLayout()
        vbox.setAlignment(Qt.AlignTop)
        self.setLayout(vbox)

        # --- Do Blur ---
        self.do_blur_button = QPushButton("Do Blur")
        self.do_blur_button.setCheckable(True)
        self.do_blur_button.clicked.connect(self.toggle_do_blur)
        self.layout().addWidget(self.do_blur_button)

        # Blur Slider
        self.blur_label = QLabel("blur_sigma")
        blur_layout = QHBoxLayout()
        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setRange(0, 30)
        self.blur_slider.setValue(10)  # Default value
        self.blur_entry = QLineEdit()
        self.blur_entry.setPlaceholderText("0 - 30 (inclusive)")
        self.blur_entry.setText(f"{self.blur_slider.value()}")
        self.blur_entry.setReadOnly(True)
        self.blur_entry.setFixedWidth(50)

        self.blur_slider.valueChanged.connect(self.update_blur)

        blur_layout.addWidget(self.blur_slider)
        blur_layout.addWidget(self.blur_entry)
        self.layout().addWidget(self.blur_label)
        self.layout().addLayout(blur_layout)

        # --- Remove Background ---
        self.remove_background_button = QPushButton("Remove Background")
        self.remove_background_button.setCheckable(True)
        self.remove_background_button.clicked.connect(self.toggle_remove_background)
        self.layout().addWidget(self.remove_background_button)

        # --- Segmentation ---
        self.segmentation_button = QPushButton("Segmentation")
        self.segmentation_button.setCheckable(True)
        self.segmentation_button.clicked.connect(self.toggle_segmentation)
        self.layout().addWidget(self.segmentation_button)

        # --- DetectionBox ---
        self.detectionbox_button = QPushButton("DetectionBox")
        self.detectionbox_button.setCheckable(True)
        self.detectionbox_button.clicked.connect(self.toggle_detectionbox)
        self.layout().addWidget(self.detectionbox_button)

        # --- Label ---
        self.label_button = QPushButton("Label")
        self.label_button.setCheckable(True)
        self.label_button.clicked.connect(self.toggle_label)
        self.layout().addWidget(self.label_button)

        # --- Darkness Slider ---
        self.darkness_label = QLabel("Darkness")
        darkness_layout = QHBoxLayout()
        self.darkness_slider = QSlider(Qt.Horizontal)
        self.darkness_slider.setRange(0, 100)
        self.darkness_slider.setValue(50)  # Default value
        self.darkness_entry = QLineEdit()
        self.darkness_entry.setPlaceholderText("0.0 - 1.0")
        self.darkness_entry.setText(f"{self.darkness_slider.value() / 100:.2f}")
        self.darkness_entry.setReadOnly(True)
        self.darkness_entry.setFixedWidth(50)
        self.darkness_slider.valueChanged.connect(self.update_darkness)
        darkness_layout.addWidget(self.darkness_slider)
        darkness_layout.addWidget(self.darkness_entry)
        self.layout().addWidget(self.darkness_label)
        self.layout().addLayout(darkness_layout)

        # Initial states
        self.segmentation_button.click()
        self.detectionbox_button.click()
        self.label_button.click()
        # Blur slider is only useful when Do Blur is on:
        self._set_blur_controls_enabled(False)

        # Emit initial state
        self.emit_state()

    # ---------- helpers ----------
    def _highlight(self, btn: QPushButton, on: bool):
        btn.setStyleSheet("background-color: #87CEEB;" if on else "")

    def _set_blur_controls_enabled(self, enabled: bool):
        self.blur_slider.setEnabled(enabled)
        self.blur_entry.setEnabled(enabled)

    def emit_state(self):
        state = {
            "do_blur": self.do_blur_button.isChecked(),
            "blur_sigma": self.blur_slider.value(),
            "remove_background": self.remove_background_button.isChecked(),
            "do_seg": self.segmentation_button.isChecked(),
            "do_det": self.detectionbox_button.isChecked(),
            "do_label": self.label_button.isChecked(),
            "darkness": self.darkness_slider.value() / 100.0,
        }
        self.value_changed.emit(state)

    # ---------- slots ----------
    def toggle_remove_background(self):
        on = self.remove_background_button.isChecked()
        self._highlight(self.remove_background_button, on)

        if on and self.do_blur_button.isChecked():
            # Enforce mutual exclusivity: turn off Do Blur
            self.do_blur_button.setChecked(False)
            self._highlight(self.do_blur_button, False)
            self._set_blur_controls_enabled(False)

        # Remove Background doesn't control blur slider availability directly,
        # but if Do Blur is off, keep slider disabled for clarity.
        if not self.do_blur_button.isChecked():
            self._set_blur_controls_enabled(False)

        self.emit_state()

    def toggle_do_blur(self):
        on = self.do_blur_button.isChecked()
        self._highlight(self.do_blur_button, on)
        self._set_blur_controls_enabled(on)

        if on and self.remove_background_button.isChecked():
            # Enforce mutual exclusivity: turn off Remove Background
            self.remove_background_button.setChecked(False)
            self._highlight(self.remove_background_button, False)

        self.emit_state()

    def toggle_segmentation(self):
        on = self.segmentation_button.isChecked()
        self._highlight(self.segmentation_button, on)
        self.emit_state()

    def toggle_detectionbox(self):
        on = self.detectionbox_button.isChecked()
        self._highlight(self.detectionbox_button, on)
        self.emit_state()

    def toggle_label(self):
        on = self.label_button.isChecked()
        self._highlight(self.label_button, on)
        self.emit_state()

    def update_darkness(self):
        darkness_value = self.darkness_slider.value() / 100
        self.darkness_entry.setText(f"{darkness_value:.2f}")
        self.emit_state()

    def update_blur(self):
        blur_value = self.blur_slider.value()
        self.blur_entry.setText(f"{blur_value}")
        self.emit_state()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = ToggleButtonWidget()
    widget.show()
    sys.exit(app.exec())
