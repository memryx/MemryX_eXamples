import os
from collections import defaultdict
from pathlib import Path
import numpy as np
import shutil

from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QWidget,
                               QVBoxLayout, QLineEdit, QPushButton,
                               QHBoxLayout, QSplitter, QCheckBox, QFrame,
                               QTreeWidget, QTreeWidgetItem, QInputDialog,
                               QMessageBox, QFileDialog)
from PySide6.QtGui import QImage, QPixmap, QMouseEvent, QKeyEvent
from PySide6.QtCore import QTimer, Qt, QThread, Signal, QMutex
import sys


def cosine_similarity(vector1, vector2):
    # Ensure the vectors are numpy arrays
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    
    # Compute the dot product and magnitudes
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    # Handle the case where the magnitude is zero to avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    # Compute cosine similarity
    cosine_sim = dot_product / (magnitude1 * magnitude2)
    
    return cosine_sim


class FaceDatabase:
    cosine_threshold = 0.48

    def __init__(self, path='../assets/db'):
        self.database = defaultdict(dict)
        self.database_path = Path(path)

    def load_database_embeddings(self, database_path):
        print(f'loading database "{database_path}"...', end='', flush=True)
        self.database = defaultdict(dict)
        self.database_path = Path(database_path)

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
                distance_dict[name].append(cosine_similarity(db_embedding, target_embedding))

        all_distances = [(name, np.max(dist)) for name, dist in distance_dict.items()]
        all_distances = sorted(all_distances, key=lambda x: x[1], reverse=True)

        if not all_distances:
            return 'Unknown', all_distances

        if all_distances[0][1] > self.cosine_threshold:
            profile_name = all_distances[0][0]

        return profile_name, all_distances


class DatabaseViewerWidget(QWidget):
    def __init__(self, face_database: FaceDatabase):
        super().__init__()

        self.face_database = face_database
        self.db_path = face_database.database_path
        self.setWindowTitle("Database Viewer")
        self.setGeometry(200, 200, 800, 600)

        # Set up layout
        self.layout_ = QVBoxLayout()
        self.setLayout(self.layout_)

        # Add button to select a different database
        self.select_db_button = QPushButton("Select Database")
        self.select_db_button.clicked.connect(self.select_database)
        self.layout_.addWidget(self.select_db_button)

        # Set up tree view
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Profiles"])
        self.tree.currentItemChanged.connect(self.on_current_item_changed)
        self.layout_.addWidget(self.tree)

        # Add button to add a new profile
        self.add_button = QPushButton("Add New Profile")
        self.add_button.clicked.connect(self.add_profile)
        self.layout_.addWidget(self.add_button)

        # Add button to delete the selected item
        self.delete_selected_button = QPushButton("Delete Selected Item")
        self.delete_selected_button.clicked.connect(self.delete_selected_item)
        self.layout_.addWidget(self.delete_selected_button)

        # Set up image preview
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.layout_.addWidget(self.preview_label)
        self.set_placeholder_image()

        # Load profiles into the tree
        self.load_profiles()

    def load_profiles(self):
        current_profile = self.get_selected_directory()

        self.tree.clear()
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
        
        self.face_database.load_database_embeddings(self.db_path)

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
        if ok:
            if profile_name:
                profile_path = os.path.join(self.db_path, profile_name)
                if not os.path.exists(profile_path):
                    os.makedirs(profile_path)
                    self.load_profiles()
                else:
                    QMessageBox.warning(self, 'Error', f"Profile '{profile_name}' already exists.")
            else:
                    QMessageBox.warning(self, 'Error', f"Enter a valid profile name.")
        return profile_name

    def delete_profile(self, profile_name):
        profile_path = os.path.join(self.db_path, profile_name)
        if os.path.exists(profile_path):
            shutil.rmtree(profile_path)
            self.load_profiles()

        self.face_database.delete_profile(profile_name)

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

                self.face_database.delete_embedding(profile_name, Path(file_path).name)

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


if __name__ == "__main__":
    app = QApplication(sys.argv)

    database = FaceDatabase() 
    db= DatabaseViewerWidget(database)
    db.show()
    sys.exit(app.exec())

