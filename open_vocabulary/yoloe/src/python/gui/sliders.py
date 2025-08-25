from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QGroupBox, QLabel, QFrame, QPushButton, QSlider, QLineEdit, QHBoxLayout
from PySide6.QtCore import Qt, Signal
import sys

class SliderWidget(QWidget):
    value_changed = Signal(dict)  # Signal to propagate slider values as a dict

    def __init__(self, conf, iou, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)

        # Confidence Slider
        self.confidence_label = QLabel("Confidence")
        confidence_layout = QHBoxLayout()
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(conf * 100)
        self.confidence_entry = QLineEdit()
        self.confidence_entry.setPlaceholderText("0.0 - 1.0")
        self.confidence_entry.setText(f"{self.confidence_slider.value() / 100:.2f}")
        self.confidence_entry.setReadOnly(True)
        self.confidence_entry.setFixedWidth(50)

        self.confidence_slider.valueChanged.connect(self.update_values)

        confidence_layout.addWidget(self.confidence_slider)
        confidence_layout.addWidget(self.confidence_entry)

        layout.addWidget(self.confidence_label)
        layout.addLayout(confidence_layout)

        # IOU Slider
        self.iou_label = QLabel("IOU")
        iou_layout = QHBoxLayout()
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(0, 100)
        self.iou_slider.setValue(iou * 100)
        self.iou_entry = QLineEdit()
        self.iou_entry.setPlaceholderText("0.0 - 1.0")
        self.iou_entry.setText(f"{self.iou_slider.value() / 100:.2f}")
        self.iou_entry.setReadOnly(True)
        self.iou_entry.setFixedWidth(50)

        self.iou_slider.valueChanged.connect(self.update_values)

        iou_layout.addWidget(self.iou_slider)
        iou_layout.addWidget(self.iou_entry)

        layout.addWidget(self.iou_label)
        layout.addLayout(iou_layout)

    def update_values(self):
        conf_value = self.confidence_slider.value() / 100
        iou_value = self.iou_slider.value() / 100
        self.confidence_entry.setText(f"{conf_value:.2f}")
        self.iou_entry.setText(f"{iou_value:.2f}")
        self.value_changed.emit({"conf": conf_value, "iou": iou_value})

if __name__ == '__main__':
    app = QApplication(sys.argv)

    widget = SliderWidget(0.15, 0.60)
    #widget.toggled.connect(lambda selected: print("Selected classes:", selected))
    widget.value_changed.connect(lambda values: print("Updated values:", values))
    widget.show()

    sys.exit(app.exec())
