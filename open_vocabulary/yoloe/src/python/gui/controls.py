from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QGroupBox, QLabel, QFrame
from PySide6.QtCore import Qt

from .list_widget import ClassEntryWidget
from .sliders import SliderWidget
from .buttons import ToggleButtonWidget

class SectionWidget(QGroupBox):
    def __init__(self, title, widget, parent=None):
        super().__init__(title, parent)
        self.setLayout(QVBoxLayout())

        # Example content inside the section
        self.widget = widget
        self.layout().addWidget(widget)

class ControlPanel(QWidget):
    def __init__(self, conf=0.15, iou=0.6, classes=['person', 'dog'], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Control Panel")
        self.setLayout(QVBoxLayout())

        self.class_entry_widget = ClassEntryWidget(classes)
        self.sliders = SliderWidget(conf, iou)
        self.buttons = ToggleButtonWidget()

        # Section 1
        self.layout().addWidget(SectionWidget("Visual Control", self.buttons))
        # self.layout().addWidget(SectionWidget("Network Config", self.sliders))
        self.layout().addWidget(SectionWidget("Class Selections", self.class_entry_widget))

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    control_panel = ControlPanel()
    control_panel.show()
    sys.exit(app.exec())

