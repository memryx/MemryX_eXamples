from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QProgressBar
from PySide6.QtCore import QTimer
import sys

class ProgressBarWidget(QWidget):
    def __init__(self, n_seconds=1, parent=None):
        super().__init__(parent)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.layout.addWidget(self.progress_bar)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)
        self.progress_value = 0

        self.update_interval = n_seconds  # Convert seconds to intervals for 100 steps

    def start(self):
        self.timer.start(self.update_interval)

    def update_progress(self):
        if self.progress_value < 100:
            self.progress_value += 1
            self.progress_bar.setValue(self.progress_value)
        else:
            self.timer.stop()

    def reset(self):
        self.timer.stop()
        self.progress_value = 0
        self.progress_bar.setValue(self.progress_value)
        #self.timer.start(self.update_interval)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    widget = ProgressBarWidget(n_seconds=5)  # Configure duration to reach 100%
    widget.show()
    widget.start()

    sys.exit(app.exec())
