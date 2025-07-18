import sys
import os
import csv
import random
from datetime import datetime
import cv2
import dlib
import numpy as np
import pyqt5
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QDesktopWidget, 
                             QLabel, QVBoxLayout, QWidget)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

class EyeTrackingDatasetGenerator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setup_camera()
        self.setup_face_detector()
        self.dataset_folder = "eye_tracking_dataset"
        os.makedirs(self.dataset_folder, exist_ok=True)
        self.csv_path = os.path.join(self.dataset_folder, "eye_tracking_data.csv")
        self.prepare_csv()

    def initUI(self):
        """Initialize the main user interface"""
        self.setWindowTitle('Eye Tracking Dataset Generator')
        self.setGeometry(100, 100, 800, 600)

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Camera preview label
        self.camera_label = QLabel('Camera Preview')
        layout.addWidget(self.camera_label)

        # Target button
        self.target_button = QPushButton('Start Tracking')
        self.target_button.clicked.connect(self.place_target)
        layout.addWidget(self.target_button)

        # Status label
        self.status_label = QLabel('Ready to track')
        layout.addWidget(self.status_label)

    def setup_camera(self):
        """Initialize camera capture"""
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            print("Error: Could not open camera.")
            sys.exit(1)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera_frame)
        self.timer.start(30)  # 30 ms interval

    def setup_face_detector(self):
        """Setup face and landmark detection"""
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    def update_camera_frame(self):
        """Capture and display camera frame"""
        ret, frame = self.camera.read()
        if ret:
            # Convert frame to RGB for display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.camera_label.setPixmap(pixmap.scaled(self.camera_label.size(), 
                                                      Qt.KeepAspectRatio))

    def place_target(self):
        """Place target button at random screen location"""
        screen_geometry = QDesktopWidget().screenNumber(QDesktopWidget().cursor().pos())
        screen_rect = QDesktopWidget().screenGeometry(screen_geometry)
        
        button_width, button_height = 50, 50
        x = random.randint(0, screen_rect.width() - button_width)
        y = random.randint(0, screen_rect.height() - button_height)

        target_window = QWidget()
        target_window.setGeometry(x, y, button_width, button_height)
        target_window.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        target_window.setStyleSheet("background-color: red;")
        
        layout = QVBoxLayout()
        target_button = QPushButton('Target')
        target_button.clicked.connect(lambda: self.record_data(x, y, target_window))
        layout.addWidget(target_button)
        target_window.setLayout(layout)
        
        target_window.show()

    def record_data(self, button_x, button_y, target_window):
        """Record eye tracking and screen data"""
        ret, frame = self.camera.read()
        if ret:
            # Detect face and landmarks
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray)
            
            if faces:
                face = faces[0]
                landmarks = self.landmark_predictor(gray, face)
                
                # Extract eye coordinates
                left_eye_x = landmarks.part(36).x
                left_eye_y = landmarks.part(36).y
                right_eye_x = landmarks.part(45).x
                right_eye_y = landmarks.part(45).y

                # Timestamp and image filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                image_filename = f"{timestamp}_eye_image.jpg"
                image_path = os.path.join(self.dataset_folder, image_filename)
                
                # Save frame
                cv2.imwrite(image_path, frame)

                # Write to CSV
                with open(self.csv_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        timestamp,
                        left_eye_x, left_eye_y,
                        right_eye_x, right_eye_y,
                        button_x, button_y,
                        image_filename
                    ])

                self.status_label.setText(f'Recorded data at {button_x}, {button_y}')
                target_window.close()
            else:
                self.status_label.setText('No face detected')

    def prepare_csv(self):
        """Prepare CSV file with headers"""
        with open(self.csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'timestamp', 
                'left_eye_x', 'left_eye_y', 
                'right_eye_x', 'right_eye_y', 
                'target_x', 'target_y', 
                'image_filename'
            ])

    def closeEvent(self, event):
        """Clean up when closing the application"""
        self.camera.release()
        event.accept()

def main():
    app = QApplication(sys.argv)
    ex = EyeTrackingDatasetGenerator()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()