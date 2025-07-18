import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time  # Import time module for delay

class GazeEstimator:
    def __init__(self):
        # Get screen resolution
        self.screen_width, self.screen_height = pyautogui.size()
        
        # MediaPipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Calibration data
        self.eye_positions = []
        self.screen_positions = []
        self.calibrated = False

    def calibrate(self, cap):
        """Perform calibration to map eye positions to screen points."""
        # 3-second delay before calibration starts
        print("Calibration will start in 3 seconds. Please prepare to focus on the dots.")

        calibration_points = [
            (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),  # Top row
            (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),  # Middle row
            (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)   # Bottom row
        ]
        print("Calibration started. Look at the green dot on the screen.")

        # Enter full-screen mode for calibration
        cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        time.sleep(3)
        for point in calibration_points:
            screen_x = int(point[0] * self.screen_width)
            screen_y = int(point[1] * self.screen_height)

            # Display calibration dot
            calibration_frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
            cv2.circle(calibration_frame, (screen_x, screen_y), 20, (0, 255, 0), -1)
            cv2.imshow('Calibration', calibration_frame)
            cv2.waitKey(1500)

            # Capture eye position
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            iris_position = self._get_iris_position(frame)
            if iris_position:
                self.eye_positions.append(iris_position)
                self.screen_positions.append((screen_x, screen_y))

        self.calibrated = True
        cv2.destroyWindow('Calibration')
        print("Calibration complete.")

    def _get_iris_position(self, frame):
        """Detect iris center."""
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Iris landmarks
                left_iris = [474, 475, 476, 477]
                right_iris = [469, 470, 471, 472]
                h, w = frame.shape[:2]

                # Process both eyes
                for iris_indices in [left_iris, right_iris]:
                    iris_points = [
                        (int(face_landmarks.landmark[idx].x * w), 
                         int(face_landmarks.landmark[idx].y * h))
                        for idx in iris_indices
                    ]
                    iris_center = (
                        int(np.mean([p[0] for p in iris_points])),
                        int(np.mean([p[1] for p in iris_points]))
                    )
                    return iris_center
        return None

    def estimate_gaze(self, frame):
        """Estimate gaze point on the screen."""
        if not self.calibrated or len(self.eye_positions) < 3:
            return frame, None
        
        # Detect current iris position
        iris_position = self._get_iris_position(frame)
        if not iris_position:
            return frame, None
        
        # Map eye position to screen using linear interpolation
        eye_x, eye_y = iris_position
        eye_positions = np.array(self.eye_positions)
        screen_positions = np.array(self.screen_positions)
        
        # Linear regression for gaze estimation
        from sklearn.linear_model import LinearRegression
        reg_x = LinearRegression().fit(eye_positions, screen_positions[:, 0])
        reg_y = LinearRegression().fit(eye_positions, screen_positions[:, 1])
        
        screen_x = int(reg_x.predict([[eye_x, eye_y]])[0])
        screen_y = int(reg_y.predict([[eye_x, eye_y]])[0])
        
        # Draw gaze point on frame
        cv2.circle(frame, (screen_x, screen_y), 20, (0, 0, 255), -1)
        return frame, (screen_x, screen_y)

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    gaze_estimator = GazeEstimator()

    # Perform calibration with delay
    gaze_estimator.calibrate(cap)

    # Enter full-screen mode for gaze tracking
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
