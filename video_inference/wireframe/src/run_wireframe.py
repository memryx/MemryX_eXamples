import copy
import time
import argparse
import numpy as np
import cv2 as cv
from queue import Queue, Full
from threading import Thread
from memryx import AsyncAccl
from PySide6 import QtWidgets, QtGui, QtCore
import sys
from collections import deque

###############################################################################
# WireframeMxa Class
###############################################################################

class WireframeMxa(QtWidgets.QApplication):
    """
    WireframeMxa class for running wireframe inference on MXA with user-defined threshold settings.
    Attributes:
        dfp_path (str): Path to the compiled DFP file.
        premodel_path (str): Path to the pre-processing model.
        postmodel_path (str): Path to the post-processing model.
        score_thr (float): Score threshold for filtering results.
        dist_thr (float): Distance threshold for filtering results.
        input_shape (list): Input shape for the model (height, width).
        done (bool): Flag to control the termination of threads.
        cap_queue (Queue): Queue for storing captured frames.
        dets_queue (Queue): Queue for storing detection results.
        vidcap (cv.VideoCapture): Video capture object for live camera feed.
        accl (AsyncAccl): Accelerator object for handling inference.
        fps (float): Current frames per second.
        fps_window (deque): Moving window to calculate average FPS.
        start_time (float): Start time for FPS calculation.
        frame_count (int): Frame count for FPS calculation.
    """

    def __init__(self, dfp_path, postmodel_path, score_thr=0.75, dist_thr=20.0):
        super().__init__(sys.argv)

        # Parameters
        self.dfp_path = dfp_path
        self.postmodel_path = postmodel_path
        self.input_shape = [512, 512]
        self.score_thr = score_thr
        self.dist_thr = dist_thr
        self.done = False
        
        # Queues
        self.cap_queue = Queue(maxsize=5)
        self.dets_queue = Queue(maxsize=5)
        
        # Video Capture
        self.vidcap = cv.VideoCapture(0)
        
        # Accelerator
        self.accl = AsyncAccl(dfp=self.dfp_path)
        self.accl.set_postprocessing_model(self.postmodel_path)
        
        self.fps = 0
        self.fps_window = deque(maxlen=100)
        self.start_time = time.time()
        self.frame_count = 0
        
        # GUI
        self.window = WireframeDisplayWindow(self)
        self.window.show()
        
        # Start the inference
        self.run()
        
###############################################################################
   
    def run(self):
        """
        Run the wireframe inference on MXA.
        Starts the display and runs the inference process by connecting inputs and outputs.
        Handles interruption and ensures a clean exit.
        """

        # Connect the input and output
        self.accl.connect_input(self.capture_and_preprocess)
        self.accl.connect_output(self.postprocess)

        # Start the PySide6 event loop
        self.exec()

        # Wait for the accelerator to finish
        try:
            self.accl.wait()
        except KeyboardInterrupt:
            print("Interrupted by user, stopping...")
        finally:
            self.done = True
            if self.accl:
                self.accl.stop()  # Stop the accelerator gracefully
            # Allow the display thread to exit naturally

###############################################################################
 
    def capture_and_preprocess(self):
        """
        Capture the video frame and preprocess it for inference.
        Captures frames from the live camera, resizes, and adds a dummy channel for processing.
        Returns:
            batch_image (np.ndarray): Preprocessed image ready for inference.
        """

        while True:
            got_frame, frame = self.vidcap.read()
            if not got_frame:
                print("Failed to read() from cv2 capture!")
                return None

            # Put the frame in the queue
            if self.cap_queue.full():
                # drop the frame and try again
                continue
            else:
                self.cap_queue.put(frame)
                self.frame_count += 1

                # Preprocess the frame
                resized_image = cv.resize(frame, (self.input_shape[0], self.input_shape[1]), interpolation=cv.INTER_AREA)
                resized_image = resized_image * 0.007843137718737125 - 1
                resized_image = np.concatenate([resized_image, np.ones([self.input_shape[0], self.input_shape[1], 1])], axis=-1)
                expanded_image = np.expand_dims(resized_image, axis=0).astype('float32')
                expanded_image = np.transpose(expanded_image, (0, 1, 2, 3))
                return expanded_image

###############################################################################

    def postprocess(self, *mxa_output):
        """
        Postprocess the output from MXA.
        Processes the output of the accelerator, calculates distances, and filters results based on thresholds.
        Args:
            *mxa_output: Variable length argument list for MXA output tensors.
        """

        pts_score, pts, vmap = [np.squeeze(output, axis=0) for output in mxa_output]

        if vmap.shape[-1] != 4:
            raise ValueError("Unexpected vmap shape: " + str(vmap.shape))
        
        start = vmap[:, :, :2]
        end = vmap[:, :, 2:]
        dist_map = np.sqrt(np.sum((start - end) ** 2, axis=-1))

        segments_list = []
        for center, score in zip(pts, pts_score):
            y, x = center
            distance = dist_map[y, x]
            if score > self.score_thr and distance > self.dist_thr:
                disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[y, x, :]
                x_start = x + disp_x_start
                y_start = y + disp_y_start
                x_end = x + disp_x_end
                y_end = y + disp_y_end
                segments_list.append([x_start, y_start, x_end, y_end, distance])

        if segments_list:
            lines = 2 * np.array(segments_list)  # 256 > 512
            lines[:, 0] = lines[:, 0] * (self.vidcap.get(cv.CAP_PROP_FRAME_WIDTH) / self.input_shape[0])
            lines[:, 1] = lines[:, 1] * (self.vidcap.get(cv.CAP_PROP_FRAME_HEIGHT) / self.input_shape[1])
            lines[:, 2] = lines[:, 2] * (self.vidcap.get(cv.CAP_PROP_FRAME_WIDTH) / self.input_shape[0])
            lines[:, 3] = lines[:, 3] * (self.vidcap.get(cv.CAP_PROP_FRAME_HEIGHT) / self.input_shape[1])
        else:
            lines =None
        
        self.dets_queue.put(lines)

###############################################################################

    def calculate_fps(self):
        """
        Calculate the FPS based on elapsed time and frame count.
        Uses a moving average window to smooth out FPS values.
        """
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        if elapsed_time > 0:
            fps = self.frame_count / elapsed_time
            self.fps_window.append(fps)
            self.fps = sum(self.fps_window) / len(self.fps_window)
        else:
            self.fps = 0
        self.start_time = current_time
        self.frame_count = 0

    def get_fps(self):
        """
        Return the current FPS.
        Returns:
            float: Current frames per second.
        """
        return self.fps

    def reset_fps(self):
        """
        Reset FPS calculation.
        Clears the moving average window and resets frame counters.
        """
        self.fps_window.clear()
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0

###############################################################################
# WireframeDisplayWindow Class
###############################################################################

class WireframeDisplayWindow(QtWidgets.QWidget):
    """
    WireframeDisplayWindow class for displaying wireframe results using PySide6.
    Attributes:
        wireframe_mxa (WireframeMxa): Instance of WireframeMxa for accessing inference data.
        timer (QTimer): Timer for periodically updating the display.
        label (QLabel): QLabel widget for displaying video frames.
        fps_label (QLineEdit): QLineEdit widget for displaying FPS.
        threshold_input (QLineEdit): QLineEdit widget for updating score threshold.
        dist_threshold_input (QLineEdit): QLineEdit widget for updating distance threshold.
        color_dropdown (QComboBox): Dropdown for selecting line color.
        line_color (tuple): Current line color for wireframe display.
    """
    def __init__(self, wireframe_mxa):
        super().__init__()
        self.setWindowTitle("Wireframe on MXA")
        self.resize(800, 500)
        
        self.wireframe_mxa = wireframe_mxa
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 // 30)
        
        # Set up QLabel for displaying frames
        self.label = QtWidgets.QLabel(self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        
        # FPS Display
        self.fps_layout = QtWidgets.QHBoxLayout()
        self.fps_label_text = QtWidgets.QLabel("FPS:")
        self.fps_label_text.setStyleSheet("font-size: 16px;")
        self.fps_label = QtWidgets.QLineEdit(self)
        self.fps_label.setReadOnly(True)
        self.fps_label.setStyleSheet("font-size: 16px; padding: 8px; border-radius: 5px;")
        self.fps_layout.addWidget(self.fps_label_text)
        self.fps_layout.addWidget(self.fps_label)
        
        # Input for score threshold
        self.threshold_layout = QtWidgets.QHBoxLayout()
        self.threshold_label = QtWidgets.QLabel("Score Threshold:")
        self.threshold_label.setStyleSheet("font-size: 16px;")
        self.threshold_input = QtWidgets.QLineEdit(self)
        self.threshold_input.setText(str(self.wireframe_mxa.score_thr))
        self.threshold_input.setStyleSheet("font-size: 16px; padding: 8px; border-radius: 5px;")
        self.threshold_input.returnPressed.connect(self.update_threshold)
        self.threshold_layout.addWidget(self.threshold_label)
        self.threshold_layout.addWidget(self.threshold_input)
        
        # Input for distance threshold
        self.dist_threshold_layout = QtWidgets.QHBoxLayout()
        self.dist_threshold_label = QtWidgets.QLabel("Distance Threshold:")
        self.dist_threshold_label.setStyleSheet("font-size: 16px;")
        self.dist_threshold_input = QtWidgets.QLineEdit(self)
        self.dist_threshold_input.setText(str(self.wireframe_mxa.dist_thr))
        self.dist_threshold_input.setStyleSheet("font-size: 16px; padding: 8px; border-radius: 5px;")
        self.dist_threshold_input.returnPressed.connect(self.update_dist_threshold)
        self.dist_threshold_layout.addWidget(self.dist_threshold_label)
        self.dist_threshold_layout.addWidget(self.dist_threshold_input)
        
        # Dropdown for wireframe color selection
        self.color_layout = QtWidgets.QHBoxLayout()
        self.color_label = QtWidgets.QLabel("Line Color:")
        self.color_label.setStyleSheet("font-size: 16px;")
        self.color_dropdown = QtWidgets.QComboBox(self)
        self.color_dropdown.addItems(["Blue", "Green", "Red", "Teal", "Yellow"])
        self.color_dropdown.setStyleSheet("font-size: 16px; padding: 8px; border-radius: 5px;")
        self.color_dropdown.currentIndexChanged.connect(self.update_color)
        self.color_layout.addWidget(self.color_label)
        self.color_layout.addWidget(self.color_dropdown)
        self.color_map = {
            "Blue": (10, 132, 255),
            "Green": (48, 209, 88),
            "Red": (255, 69, 58),
            "Teal": (172, 182, 77),
            "Yellow": (255, 214, 10)
        }
        self.line_color = self.color_map["Blue"][::-1]  # Convert BGR to RGB for OpenCV
        
        self.backend_label = QtWidgets.QLabel("Backend:")
        self.backend_label.setStyleSheet("font-size: 16px;")
                
        # Video GroupBox
        video_groupbox = QtWidgets.QGroupBox("live Camera")
        video_layout = QtWidgets.QVBoxLayout()
        video_layout.addWidget(self.label)
        video_groupbox.setLayout(video_layout)

        info_groupbox = QtWidgets.QGroupBox("Info")
        info_groupbox_layout = QtWidgets.QVBoxLayout()
        info_groupbox_layout.addLayout(self.fps_layout)
        info_groupbox.setLayout(info_groupbox_layout)        
        # Input for video source
        
        # Model Controls GroupBox
        model_controls_groupbox = QtWidgets.QGroupBox("Model Controls")
        model_controls_layout = QtWidgets.QVBoxLayout()
        model_controls_layout.addLayout(self.threshold_layout)
        model_controls_layout.addLayout(self.dist_threshold_layout)
        model_controls_layout.addLayout(self.color_layout)
        model_controls_layout.addStretch()
        model_controls_groupbox.setLayout(model_controls_layout)
        # Sidebar layout
        sidebar_layout = QtWidgets.QVBoxLayout()
        sidebar_layout.addWidget(info_groupbox)
        sidebar_layout.addWidget(model_controls_groupbox)
        
        # Main layout
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addWidget(video_groupbox, stretch=3)
        main_layout.addLayout(sidebar_layout, stretch=1)
        self.setLayout(main_layout)

    def closeEvent(self, event):
        """
        Handle window close event to ensure a clean exit.
        Stops the timer and sets the done flag to terminate threads gracefully.
        Args:
            event (QCloseEvent): The close event triggered by the user.
        """
        self.wireframe_mxa.done = True
        self.timer.stop()
        if self.wireframe_mxa.accl:
            self.wireframe_mxa.accl.stop()  # Stop the accelerator gracefully
        self.wireframe_mxa.vidcap.release()  # Release the video capture
        event.accept()

###############################################################################

    def update_threshold(self):
        """
        Update the score threshold based on user input.
        Reads the user input from the threshold input field and updates the score threshold.
        """
        try:
            new_threshold = float(self.threshold_input.text())
            if 0 <= new_threshold <= 1:
                self.wireframe_mxa.score_thr = new_threshold
            else:
                QtWidgets.QMessageBox.warning(self, "Invalid Input", "Threshold must be between 0 and 1.")
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please enter a numeric value.")

###############################################################################

    def update_dist_threshold(self):
        """
        Update the distance threshold based on user input.
        Reads the user input from the distance threshold input field and updates the distance threshold.
        """
        try:
            new_dist_threshold = float(self.dist_threshold_input.text())
            if new_dist_threshold >= 0:
                self.wireframe_mxa.dist_thr = new_dist_threshold
            else:
                QtWidgets.QMessageBox.warning(self, "Invalid Input", "Distance threshold must be non-negative.")
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please enter a numeric value.")

###############################################################################

    def update_color(self):
        """
        Update the line color based on the user's selection from the dropdown.
        Converts the selected color to RGB format for use with OpenCV.
        """
        color_name = self.color_dropdown.currentText()
        self.line_color = self.color_map[color_name][::-1]  # Convert BGR to RGB for OpenCV

###############################################################################

    def update_frame(self):
        """
        Update the displayed video frame with wireframe overlays.
        Retrieves the latest frame and detections, draws the wireframe, and updates the QLabel.
        """
        if not self.wireframe_mxa.done:
            try:
                frame = self.wireframe_mxa.cap_queue.get()
                lines = self.wireframe_mxa.dets_queue.get()
                if lines is not None:
                    for line in lines:
                        x_start, y_start, x_end, y_end, distance = line
                        x_start, y_start, x_end, y_end = map(int, [x_start, y_start, x_end, y_end])
                        cv.line(frame, (x_start, y_start), (x_end, y_end), self.line_color, 2, cv.LINE_AA)
                
                self.wireframe_mxa.calculate_fps()
                fps = self.wireframe_mxa.get_fps()
                self.fps_label.setText(f"{int(round(fps))}")
                
                # Convert frame to QImage and set to QLabel
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_img = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888).rgbSwapped()
                self.label.setPixmap(QtGui.QPixmap.fromImage(q_img))
            except Exception as e:
                print(f"Error in updating frame: {e}")

###############################################################################
# Main
###############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run MXA real-time inference with options for DFP file, pre-processing model, and post model file path.")
    parser.add_argument('-d', '--dfp', type=str, default='../models/M_LSD_512_512_4_tflite.dfp', help="Specify the path to the compiled DFP file.")
    parser.add_argument('--postmodel', type=str, default='../models/M_LSD_512_512_4_tflite_post.tflite', help="Specify the path to the post-processing model.")
    args = parser.parse_args()
    wireframe_mxa = WireframeMxa(
        dfp_path=args.dfp,
        postmodel_path=args.postmodel
    )
  
