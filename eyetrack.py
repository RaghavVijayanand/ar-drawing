import cv2
import numpy as np
import mediapipe as mp
import pyautogui

class IrisDetector:
    def __init__(self):
        # Get screen resolution
        self.screen_width, self.screen_height = pyautogui.size()
        
        # MediaPipe face mesh for eye landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect_iris(self, frame):
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Run face mesh detection
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Iris detection indices
                left_iris = [474, 475, 476, 477]
                right_iris = [469, 470, 471, 472]
                
                # Process both eyes
                for iris_indices in [left_iris, right_iris]:
                    # Get iris landmarks
                    h, w = frame.shape[:2]
                    iris_points = [
                        (int(face_landmarks.landmark[idx].x * w), 
                         int(face_landmarks.landmark[idx].y * h)) 
                        for idx in iris_indices
                    ]
                    
                    # Compute iris center and radius
                    center_x = int(np.mean([p[0] for p in iris_points]))
                    center_y = int(np.mean([p[1] for p in iris_points]))
                    
                    # Estimate radius (distance between first and third point)
                    radius = int(np.linalg.norm(
                        np.array(iris_points[0]) - np.array(iris_points[2])
                    ) / 2)
                    
                    # Draw iris
                    cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 2)
                    cv2.circle(frame, (center_x, center_y), 2, (0, 0, 255), -1)
                    
                    # Optional: Map iris position to screen
                    screen_x = int((center_x / w) * self.screen_width)
                    screen_y = int((center_y / h) * self.screen_height)
                    
                    # Optional: Move cursor (commented out by default)
                    # pyautogui.moveTo(screen_x, screen_y)
        
        return frame

def main():
    # Create full-screen window
    cv2.namedWindow('Iris Detection', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Iris Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Create iris detector
    detector = IrisDetector()
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Resize frame to full screen
        frame = cv2.resize(frame, (pyautogui.size()[0], pyautogui.size()[1]))
        
        # Detect and draw iris
        frame_with_iris = detector.detect_iris(frame)
        
        # Show frame
        cv2.imshow('Iris Detection', frame_with_iris)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()