import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class TransferLearningGazeEstimator:
    def __init__(self):
        # Screen setup
        self.screen_width, self.screen_height = pyautogui.size()
        
        # MediaPipe face mesh setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Transfer learning model setup
        self.feature_extractor = None
        self.gaze_model = None
        self.feature_scaler_x = StandardScaler()
        self.feature_scaler_y = StandardScaler()
        
        # Calibration data
        self.calibration_data = {
            'features': [],
            'screen_x': [],
            'screen_y': []
        }
        self.calibrated = False
        
        # Smoothing
        self.gaze_history = []
        self.HISTORY_LENGTH = 5

    def _extract_face_image(self, frame):
        """Extract face region for transfer learning"""
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            
            # Compute face bounding box
            x_coords = [int(lm.x * w) for lm in landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in landmarks.landmark]
            
            x_min, x_max = max(0, min(x_coords)), min(w, max(x_coords))
            y_min, y_max = max(0, min(y_coords)), min(h, max(y_coords))
            
            # Extract face region
            face_img = frame[y_min:y_max, x_min:x_max]
            
            # Resize and preprocess for MobileNetV2
            face_img_resized = cv2.resize(face_img, (224, 224))
            face_img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(
                np.expand_dims(face_img_resized, axis=0)
            )
            
            return face_img_preprocessed
        return None

    def _create_transfer_learning_model(self):
        """Create transfer learning model using MobileNetV2"""
        # Load pre-trained MobileNetV2 without top layers
        base_model = MobileNetV2(
            weights='imagenet', 
            include_top=False, 
            input_shape=(224, 224, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom layers for gaze estimation
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        # Separate output heads for X and Y coordinates
        x_output = Dense(1, name='x_coordinate')(x)
        y_output = Dense(1, name='y_coordinate')(x)
        
        # Create model
        model = Model(
            inputs=base_model.input, 
            outputs=[x_output, y_output]
        )
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss={
                'x_coordinate': 'mse',
                'y_coordinate': 'mse'
            },
            metrics={
                'x_coordinate': 'mae',
                'y_coordinate': 'mae'
            }
        )
        
        return model

    def calibrate(self, cap):
        """Advanced calibration with transfer learning"""
        print("Calibration will start in 3 seconds. Prepare to focus on dots.")
        time.sleep(3)

        # Create transfer learning model
        self.gaze_model = self._create_transfer_learning_model()

        # 5x5 calibration grid
        grid_points = [
            (x/4, y/4) for y in range(5) for x in range(5)
        ]

        cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Collect calibration data
        for point in grid_points:
            # Compute screen coordinates
            screen_x = int(point[0] * self.screen_width)
            screen_y = int(point[1] * self.screen_height)

            # Display calibration dot
            calibration_frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
            cv2.circle(calibration_frame, (screen_x, screen_y), 20, (0, 255, 0), -1)
            cv2.imshow('Calibration', calibration_frame)
            cv2.waitKey(1000)  # Hold for 1 second

            # Capture frame and extract face image
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            face_img = self._extract_face_image(frame)

            if face_img is not None:
                self.calibration_data['features'].append(face_img[0])
                self.calibration_data['screen_x'].append(screen_x)
                self.calibration_data['screen_y'].append(screen_y)

        cv2.destroyWindow('Calibration')

        # Prepare data for training
        X = np.array(self.calibration_data['features'])
        y_x = np.array(self.calibration_data['screen_x'])
        y_y = np.array(self.calibration_data['screen_y'])

        # Split data
        X_train, X_test, y_train_x, y_test_x, y_train_y, y_test_y = train_test_split(
            X, y_x, y_y, test_size=0.2, random_state=42
        )

        # Train model
        self.gaze_model.fit(
            X_train, 
            {'x_coordinate': y_train_x, 'y_coordinate': y_train_y},
            validation_data=(X_test, {'x_coordinate': y_test_x, 'y_coordinate': y_test_y}),
            epochs=50,
            batch_size=8
        )

        self.calibrated = True
        print("Calibration and training complete.")

    def estimate_gaze(self, frame):
        """Estimate gaze point using transfer learning model"""
        if not self.calibrated:
            return frame, None

        # Extract face image
        face_img = self._extract_face_image(frame)
        if face_img is None:
            return frame, None

        # Predict coordinates
        x_pred, y_pred = self.gaze_model.predict(face_img)
        screen_x = int(x_pred[0][0])
        screen_y = int(y_pred[0][0])

        # Apply smoothing
        self.gaze_history.append((screen_x, screen_y))
        if len(self.gaze_history) > self.HISTORY_LENGTH:
            self.gaze_history.pop(0)

        smoothed_x = int(np.mean([x[0] for x in self.gaze_history]))
        smoothed_y = int(np.mean([y[1] for y in self.gaze_history]))

        # Draw gaze point
        cv2.circle(frame, (smoothed_x, smoothed_y), 20, (0, 0, 255), -1)
        return frame, (smoothed_x, smoothed_y)

def main():
    # Check GPU availability
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    gaze_estimator = TransferLearningGazeEstimator()

    # Calibration
    gaze_estimator.calibrate(cap)

    # Fullscreen tracking
    cv2.namedWindow('Gaze Tracking', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Gaze Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Gaze tracking started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame, gaze_point = gaze_estimator.estimate_gaze(frame)

        if gaze_point:
            print(f"Gaze point: {gaze_point}")

        cv2.imshow('Gaze Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()