import time
import numpy as np
from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QWidget, QSlider,
                               QVBoxLayout, QLineEdit, QPushButton,
                               QHBoxLayout, QSplitter, QCheckBox, QFrame,
                               QTreeWidget, QTreeWidgetItem, QInputDialog,
                               QMessageBox, QFileDialog)
from PySide6.QtGui import QImage, QPixmap, QMouseEvent, QKeyEvent
from PySide6.QtCore import QTimer, Qt, QThread, Signal, QMutex

class Framerate:
    def __init__(self, window=30):
        self.timestamps = [0] * window

    def update(self):
        self.timestamps = self.timestamps[1:] + [time.time()]

    def reset(self):
        self.timestamps = [0] * len(self.timestamps)

    def get(self):
        if not all(self.timestamps):
            return -1
        ts = np.array(self.timestamps)
        return 1 / np.average(ts[1:] - ts[:-1])

class SliderWithLabel(QWidget):
    """A composite widget with a label, slider, and line edit.
    
    The slider value is internally an integer, but a multiplier converts it
    to a float value for use (e.g. scale factor). The widget emits valueChanged
    signals as floats.
    """
    valueChanged = Signal(float)
    
    def __init__(self, 
                 label_text, 
                 orientation=Qt.Horizontal, 
                 minimum=0,
                 maximum=100, 
                 initial=50, 
                 step=1, 
                 multiplier=1.0, 
                 parent=None):
        super().__init__(parent)
        self.multiplier = multiplier
        
        self.label = QLabel(label_text)
        self.slider = QSlider(orientation)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(initial)
        self.slider.setSingleStep(step)
        
        # Display the actual (multiplied) value.
        init_value = initial * self.multiplier
        self.lineEdit = QLineEdit(f"{init_value:.2f}" if multiplier != 1 else str(initial))
        self.lineEdit.setFixedWidth(50)
        
        layout = QHBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        layout.addWidget(self.lineEdit)
        
        self.slider.valueChanged.connect(self.onSliderChanged)
        self.lineEdit.editingFinished.connect(self.onLineEditChanged)
    
    def onSliderChanged(self, value):
        float_value = value * self.multiplier
        if self.multiplier != 1:
            self.lineEdit.setText(f"{float_value:.2f}")
        else:
            self.lineEdit.setText(str(value))
        self.valueChanged.emit(float_value)
    
    def onLineEditChanged(self):
        try:
            float_value = float(self.lineEdit.text())
        except ValueError:
            float_value = self.slider.value() * self.multiplier
        int_value = int(round(float_value / self.multiplier))
        self.slider.setValue(int_value)
        self.valueChanged.emit(float_value)
    
    def value(self):
        return self.slider.value() * self.multiplier
    
    def setValue(self, value):
        int_value = int(round(value / self.multiplier))
        self.slider.setValue(int_value)

