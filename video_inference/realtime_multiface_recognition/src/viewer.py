import os
import sys
import shutil
import queue
import time
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QWidget,
                               QVBoxLayout, QLineEdit, QPushButton,
                               QHBoxLayout, QSplitter, QCheckBox, QFrame,
                               QTreeWidget, QTreeWidgetItem, QInputDialog,
                               QMessageBox, QFileDialog)
from PySide6.QtGui import QImage, QPixmap, QMouseEvent, QKeyEvent
from PySide6.QtCore import QTimer, Qt, QThread, Signal, QMutex

mx_face_path = Path("../../../image_inference/face_recognition/src")
sys.path.append(str(mx_face_path))
from MXFace import MXFace

FORCE_1080P_30FPS = True

# Global flag for thread termination
#pipeline_lock = QMutex()
stop_threads = False
mx_face = MXFace("../models/")

class FaceDatabase:
    def __init__(self):
        self.database = defaultdict(dict)

    def load_database_embeddings(self, database_path):
        print(f'loading database "{database_path}"...', end='', flush=True)
        self.database = defaultdict(dict)

        embedding_paths = []
        # Walk through directory recursively
        for root, dirs, files in os.walk(database_path):
            for file in files:
                if file.lower().endswith('embed'):
                    # Full path to the image
                    embed = np.loadtxt(os.path.join(root, file))
                    name = Path(root).name
                    self.database[name][file] = embed
        print(f'Done.')

    def delete_profile(self, profile_name):
        if profile_name in self.database:
            self.database.pop(profile_name)

    def delete_embedding(self, profile_name, embedding_file_name):
        if embedding_file_name in self.database[profile_name]:
            self.database[profile_name].pop(embedding_file_name)

    def add_to_database(self, embedding, profile_image_path):
        # Save embedding 
        embed_path = profile_image_path.replace('.jpg', '.embed')
        np.savetxt(f'{embed_path}', embedding)

        # Update database
        file_name = Path(embed_path).name
        profile = embed_path.split('/')[-2]
        self.database[profile][file_name] = embedding

    def find(self, target_embedding):
        profile_name, max_distance = 'Unknown', float('-inf')

        all_distances = []
        all_hits = []
        distance_dict = defaultdict(list)
        for name, db_embeddings in self.database.items():
            distances = []
            if not db_embeddings:
                continue

            for (file_name, db_embedding) in db_embeddings.items():
                distance_dict[name].append(mx_face.cosine_similarity(db_embedding, target_embedding))

        all_distances = [(name, np.max(dist)) for name, dist in distance_dict.items()]
        all_distances = sorted(all_distances, key=lambda x: x[1], reverse=True)

        if not all_distances:
            return 'Unknown', all_distances

        if all_distances[0][1] > mx_face.cosine_threshold:
            profile_name = all_distances[0][0]

        return profile_name, all_distances

face_database = FaceDatabase()

# Thread for reading video frames
class VideoReaderThread(QThread):
    frame_ready = Signal(np.ndarray)

    def __init__(self, video_source, frame_queue):
        super().__init__()
        self.video_source = video_source
        self.frame_queue = frame_queue
        self.stop_threads = False
        self.pause = False
        self.cur_frame = None

    def toggle_play(self):
        self.pause = not self.pause

    def run(self):
        if self.video_source.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Handle static image case
            frame = cv2.imread(self.video_source)
            if frame is None:
                print("Failed to load image.")
                return
            while not self.stop_threads:
                if not self.frame_queue.full():
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.frame_queue.put(np.array(rgb_frame))
                time.sleep(1 / 30)  # Simulate 30fps for static image
        else:
            # Handle video case
            cap = cv2.VideoCapture(self.video_source)
            if FORCE_1080P_30FPS:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FPS, 30)

            while not self.stop_threads:
                if self.pause:
                    if self.cur_frame is not None:
                        try:
                            self.frame_queue.put(np.array(rgb_frame), block=False)
                        except queue.Full:
                            pass
                    time.sleep(0.1)
                    continue

                ret, frame = cap.read()
                if not ret:
                    if self.video_source.endswith('.mp4') or self.video_source.endswith('.webm'):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    else:
                        print("Stream ended or failed to grab frame.")
                        break
                else:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.cur_frame = np.array(rgb_frame)

                    if self.video_source.endswith('.mp4') or self.video_source.endswith('.webm'):
                        start = time.time()
                        self.frame_queue.put(np.array(rgb_frame))
                        dt = time.time() - start
                        time.sleep(max(0.033-dt, 0))  # Simulating real-time video stream (30fps)
                    else:
                        #if not self.frame_queue.full():
                            #rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            #self.frame_queue.put(np.array(rgb_frame))
                        try:
                            #if pipeline_lock.try_lock():
                            self.frame_queue.put(self.cur_frame, timeout=0.033)
                        except queue.Full:
                            print('Dropped Frame')

            cap.release()
            print("Video reader stopped.")

    def stop(self):
        self.stop_threads = True

# Thread for displaying video frames
class VideoDisplayThread(QThread):
    frame_ready = Signal(np.ndarray)

    def __init__(self, frame_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.stop_threads = False

    def run(self):
        while not self.stop_threads or not self.frame_queue.empty():
            try:
                annotated_frame = self.frame_queue.get(timeout=1)  # Timeout to allow shutdown
                self.frame_ready.emit(annotated_frame)
            except queue.Empty:
                print('Unable to get.. skipping')
                continue
        print("Video display stopped.")

    def stop(self):
        self.stop_threads = True

class DatabaseViewer(QWidget):
    def __init__(self, db_path="db"):
        super().__init__()

        self.db_path = db_path
        self.setWindowTitle("Database Viewer")
        self.setGeometry(200, 200, 800, 600)

        # Set up layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Add button to select a different database
        self.select_db_button = QPushButton("Select Database")
        self.select_db_button.clicked.connect(self.select_database)
        self.layout.addWidget(self.select_db_button)

        # Set up tree view
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Profiles"])
        self.tree.currentItemChanged.connect(self.on_current_item_changed)
        self.layout.addWidget(self.tree)

        # Add button to add a new profile
        self.add_button = QPushButton("Add New Profile")
        self.add_button.clicked.connect(self.add_profile)
        self.layout.addWidget(self.add_button)

        # Add button to delete the selected item
        self.delete_selected_button = QPushButton("Delete Selected Item")
        self.delete_selected_button.clicked.connect(self.delete_selected_item)
        self.layout.addWidget(self.delete_selected_button)

        # Set up image preview
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.preview_label)
        self.set_placeholder_image()

        # Load profiles into the tree
        self.load_profiles()

    def load_profiles(self):
        current_profile = self.get_selected_directory()

        self.tree.clear()
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
        
        face_database.load_database_embeddings(self.db_path)

        # Iterate over profiles in the database directory
        for profile_name in os.listdir(self.db_path):
            profile_path = os.path.join(self.db_path, profile_name)
            if os.path.isdir(profile_path):
                profile_item = QTreeWidgetItem([profile_name])
                self.tree.addTopLevelItem(profile_item)

                # Add .jpg files under each profile
                for file_name in os.listdir(profile_path):
                    if file_name.endswith(".jpg"):
                        file_item = QTreeWidgetItem([file_name])
                        profile_item.addChild(file_item)

                # By default, collapse the profile item
                if profile_path == current_profile:
                    self.tree.expandItem(profile_item)
                    self.tree.setCurrentItem(profile_item)
                else:
                    self.tree.collapseItem(profile_item)
        
    def on_current_item_changed(self, current, previous):
        if current:
            if not current.parent():  # Top-level item (Profile)
                # Collapse all profiles
                for i in range(self.tree.topLevelItemCount()):
                    profile_item = self.tree.topLevelItem(i)
                    self.tree.collapseItem(profile_item)
                # Expand the currently selected profile
                self.tree.expandItem(current)

                # Preview the first JPEG in the profile
                profile_name = current.text(0)
                profile_path = os.path.join(self.db_path, profile_name)
                for file_name in os.listdir(profile_path):
                    if file_name.endswith(".jpg"):
                        self.preview_image(os.path.join(profile_path, file_name))
                        return
                self.set_placeholder_image()
            else:  # Child item (.jpg file)
                profile_name = current.parent().text(0)
                file_name = current.text(0)
                file_path = os.path.join(self.db_path, profile_name, file_name)
                self.preview_image(file_path)
        else:
            self.set_placeholder_image()

    def preview_image(self, image_path):
        pixmap = QPixmap(image_path)

        if not pixmap.isNull():
            self.preview_label.setPixmap(pixmap.scaled(224, 224, Qt.KeepAspectRatio))
        else:
            self.set_placeholder_image()

    def set_placeholder_image(self):
        # Create a black placeholder image
        placeholder = QImage(224, 224, QImage.Format_RGB32)
        placeholder.fill(Qt.black)
        pixmap = QPixmap.fromImage(placeholder)
        self.preview_label.setPixmap(pixmap)

    def add_profile(self):
        profile_name, ok = QInputDialog.getText(self, 'Add New Profile', 'Enter profile name:')
        if ok and profile_name:
            profile_path = os.path.join(self.db_path, profile_name)
            if not os.path.exists(profile_path):
                os.makedirs(profile_path)
                self.load_profiles()
            else:
                QMessageBox.warning(self, 'Error', f"Profile '{profile_name}' already exists.")
        return profile_name

    def delete_profile(self, profile_name):
        profile_path = os.path.join(self.db_path, profile_name)
        if os.path.exists(profile_path):
            shutil.rmtree(profile_path)
            self.load_profiles()

        face_database.delete_profile(profile_name)

    def delete_selected_item(self):
        selected_item = self.tree.currentItem()
        #next_item = self.tree.itemAbove(selected_item)

        if selected_item:
            parent = selected_item.parent()
            if parent is None:  # Top-level item (Profile)
                profile_name = selected_item.text(0)
                self.delete_profile(profile_name)
            else:  # Child item (.jpg file)
                profile_name = parent.text(0)
                file_name = selected_item.text(0)
                file_path = os.path.join(self.db_path, profile_name, file_name)
                # Delete JPG image
                if os.path.exists(file_path):
                    os.remove(file_path)
                    parent.removeChild(selected_item)

                # Delete the embedding
                file_path = file_path.replace('.jpg', '.embed')
                if os.path.exists(file_path):
                    os.remove(file_path)

                face_database.delete_embedding(profile_name, Path(file_path).name)

    def select_database(self):
        new_db_path = QFileDialog.getExistingDirectory(self, "Select Database Directory", "./")
        if new_db_path:
            self.db_path = new_db_path
            self.load_profiles()

    def get_selected_directory(self):
        selected_item = self.tree.currentItem()
        if selected_item:
            parent = selected_item.parent()
            if parent is None:  # Top-level item (Profile)
                profile_name = selected_item.text(0)
                return os.path.join(self.db_path, profile_name)
            else:  # Child item (.jpg file)
                profile_name = parent.text(0)
                return os.path.join(self.db_path, profile_name)
        return None

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Delete:
            self.delete_selected_item()
        if event.key() == Qt.Key_Escape:
            self.tree.setCurrentItem(None)
        else:
            super().keyPressEvent(event)

class VideoPlayer(QMainWindow):
    def __init__(self, video_path='/dev/video0', db_path='db/'):
        super().__init__()
        self.video_path = video_path
        self.setWindowTitle("Video Player Loop")
        self.resize(1200, 800)

        # Set up main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setMinimumSize(300, 200)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Splitter to separate control panel and video viewer
        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)

        # Control panel widget
        self.control_panel = QWidget()
        self.control_panel.setFixedWidth(300)
        self.control_layout = QVBoxLayout(self.control_panel)
        self.splitter.addWidget(self.control_panel)

        # Video path input and load button
        self.video_path_input = QLineEdit(self)
        self.video_path_input.setPlaceholderText("Enter video file path...")
        self.load_video_button = QPushButton("Load Video Path", self)
        self.load_video_button.clicked.connect(self.update_video_path)
        self.control_layout.addWidget(self.video_path_input)
        self.control_layout.addWidget(self.load_video_button)

        # Config panel with checkboxes
        self.config_panel = QFrame()
        self.config_layout = QVBoxLayout(self.config_panel)
        self.keypoints_checkbox = QCheckBox("Draw Keypoints", self)
        self.keypoints_checkbox.setChecked(False)
        self.bbox_checkbox = QCheckBox("Draw Boxes", self)
        self.bbox_checkbox.setChecked(True)
        self.conf_checkbox = QCheckBox("Show Distances", self)
        self.conf_checkbox.setChecked(False)
        self.config_layout.addWidget(self.keypoints_checkbox)
        self.config_layout.addWidget(self.bbox_checkbox)
        self.config_layout.addWidget(self.conf_checkbox)
        self.control_layout.addWidget(self.config_panel)

        # Database loader
        self.database_viewer = DatabaseViewer(db_path=db_path)
        self.control_layout.addWidget(self.database_viewer)

        # Video viewer widget
        self.video_widget = QWidget()
        self.video_layout = QVBoxLayout(self.video_widget)
        self.splitter.addWidget(self.video_widget)
        self.splitter.setStretchFactor(1, 1)

        # Video display label
        self.video_label = QLabel(self)
        self.video_label.mousePressEvent = self.handle_mouse_click
        self.video_label.setMouseTracking(True)
        self.video_label.mouseMoveEvent = self.handle_mouse_move
        self.video_layout.addWidget(self.video_label)

        # Set up video-related attributes
        self.frame_queue = mx_face
        self.video_reader_thread = VideoReaderThread(self.video_path, self.frame_queue)
        self.video_display_thread = VideoDisplayThread(self.frame_queue)

        # Connect signals to slots
        self.video_reader_thread.frame_ready.connect(self.update_frame)
        self.video_display_thread.frame_ready.connect(self.update_frame)

        # Start threads
        self.video_reader_thread.start()
        self.video_display_thread.start()

        self.current_frame = None
        self.annotated_frame = None
        self.mouse_position = None
        self.timestamps = [0] * 30

    def update_video_path(self):
        # Update the video path and reload the video
        new_path = self.video_path_input.text()
        if new_path:
            self.video_path = new_path
            self.video_reader_thread.stop()
            self.video_reader_thread.wait()
            self.video_reader_thread = VideoReaderThread(self.video_path, self.frame_queue)
            self.video_reader_thread.frame_ready.connect(self.update_frame)
            self.video_reader_thread.start()

    def update_frame(self, annotated_frame):
        cur_time = time.time()
        self.timestamps.append(cur_time)
        self.timestamps.pop(0)
        dt = np.average([self.timestamps[i + 1] - self.timestamps[i] for i in range(len(self.timestamps) - 1)])

        self.current_frame = annotated_frame.image
        self.annotated_frame = annotated_frame

        # Draw bounding boxes and labels for each face in the frame
        frame = annotated_frame.image.copy()
        fps_label = f'FPS:{1 / dt:.1f}'
        cv2.putText(frame, fps_label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 0), 2)
        for face in annotated_frame.detected_faces:
            (left, top, width, height) = face.bbox

            profile_name, distance_list = face_database.find(face.embedding)
            face.person_id = profile_name

            label = f'{face.person_id}'
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
            if self.conf_checkbox.isChecked():
                for i, (name, distance) in enumerate(distance_list):
                    if i == 3:
                        break
                    label = f'{name}: {distance:.1f}'
                    cv2.putText(frame, label, (left + 10, top + 10 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

            if self.bbox_checkbox.isChecked():
                cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)

            # Draw keypoints if checkbox is checked
            if self.keypoints_checkbox.isChecked():
                for (x, y) in face.keypoints:
                    cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

            # Draw semi-transparent rectangle if mouse is inside bounding box
            if self.mouse_position:
                mouse_x, mouse_y = self.mouse_position
                if left <= mouse_x <= left + width and top <= mouse_y <= top + height:
                    overlay = frame.copy()
                    alpha = 0.5  # Transparency factor
                    cv2.rectangle(overlay, (left, top), (left + width, top + height), (0, 0, 255), -1)
                    # Apply the overlay
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

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

        # Get image information
        height, width, channels = frame.shape
        bytes_per_line = channels * width

        # Create QImage and display it
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def handle_mouse_click(self, event: QMouseEvent):
        if self.current_frame is None or self.annotated_frame is None:
            return

        # Get the position of the mouse click relative to the QLabel
        mouse_x = event.position().x()
        mouse_y = event.position().y()

        # Calculate the scaling factors
        video_label_width = self.video_label.width()
        video_label_height = self.video_label.height()
        frame_height, frame_width, _ = self.current_frame.shape

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

            # Iterate over each face to check if the click is inside any bounding box
            found = False 
            for face in self.annotated_frame.detected_faces:
                (left, top, width, height) = face.bbox
                if left <= adjusted_x <= left + width and top <= adjusted_y <= top + height:
                    # Make the bounding box square with a 10px margin on all sides
                    margin = 10
                    bbox_size = max(width, height) + 2 * margin
                    center_x, center_y = left + width // 2, top + height // 2
                    x_start = max(0, center_x - bbox_size // 2)
                    x_end = min(self.current_frame.shape[1], center_x + bbox_size // 2)
                    y_start = max(0, center_y - bbox_size // 2)
                    y_end = min(self.current_frame.shape[0], center_y + bbox_size // 2)

                    # Crop the frame
                    cropped_frame = self.current_frame[y_start:y_end, x_start:x_end]

                    # Save the cropped image as a jpg file in the selected directory
                    profile_path = self.database_viewer.get_selected_directory()
                    if not profile_path:
                        if face.person_id == 'Unknown':
                            new_profile = self.database_viewer.add_profile()
                            profile_path = os.path.join(self.database_viewer.db_path, new_profile)
                        else:
                            profile_path = os.path.join(self.database_viewer.db_path, face.person_id)

                    if os.path.exists(profile_path):
                        i = 0
                        while os.path.exists(os.path.join(profile_path, f"{i}.jpg")):
                            i += 1
                        filename = os.path.join(profile_path, f"{i}.jpg")

                        print(f'Saving image to {filename}')
                        cv2.imwrite(filename, cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR))
                        self.database_viewer.load_profiles()
                        #face_database.add_to_database(cropped_frame, filename)
                        face_database.add_to_database(face.embedding, filename)
                    found = True
                    break

            if found == False:
                self.video_reader_thread.toggle_play()

    def handle_mouse_move(self, event: QMouseEvent):
        if self.current_frame is None:
            return

        # Get the position of the mouse relative to the QLabel
        mouse_x = event.position().x()
        mouse_y = event.position().y()

        # Calculate the scaling factors
        video_label_width = self.video_label.width()
        video_label_height = self.video_label.height()
        frame_height, frame_width, _ = self.current_frame.shape

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
            self.mouse_position = (adjusted_x, adjusted_y)
        else:
            self.mouse_position = None

    def closeEvent(self, event):
        # Stop threads and release video capture on close
        self.video_reader_thread.stop()
        self.video_reader_thread.wait()
        self.video_display_thread.stop()
        self.video_display_thread.wait()
        self.frame_queue.stop()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    video_path = "/dev/video0"  # Replace with your video file path
    db_path = "db/"  # Replace with your video file path
    #video_path = 'sources/friends.webm' #'/dev/video0'
    #db_path = 'friends_db' #'db'
    player = VideoPlayer(video_path, db_path)
    player.show()
    sys.exit(app.exec())
