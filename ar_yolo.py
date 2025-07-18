import cv2
import numpy as np
import torch
import mediapipe as mp
from math import sqrt
import time
from threading import Thread, Lock
from queue import Queue, Empty
import sys
import os
import traceback

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class YOLODetector:
    def __init__(self):
        # Load YOLO model from PyTorch Hub
        self.model = torch.hub.load(
            'ultralytics/yolov5',
            'yolov5s',
            pretrained=True,
            trust_repo=True
        )
        self.model.to(device)
        self.model.eval()
        
        # Lower detection thresholds for better sensitivity in AR mode
        self.model.conf = 0.2  # confidence threshold
        self.model.iou = 0.45  # non-maximum suppression IoU threshold
        
        # Detection thread
        self.detect_queue = Queue(maxsize=1)
        self.detect_result = None
        self.lock = Lock()
        self.running = True
        
        # Start detection thread
        self.detect_thread = Thread(target=self._detect_worker)
        self.detect_thread.daemon = True
        self.detect_thread.start()
    
    def _detect_worker(self):
        """Worker thread to run object detection asynchronously."""
        while self.running:
            # Pull a frame for detection, skip if none
            try:
                frame = self.detect_queue.get(timeout=0.1)
            except Empty:
                continue
            # Convert BGR to RGB for the model
            try:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"Error converting frame to RGB: {e}")
                traceback.print_exc()
                continue
            # Run YOLO inference
            try:
                results = self.model(img)
                # Store results
                with self.lock:
                    self.detect_result = results
            except Exception as e:
                print(f"Detection error: {repr(e)}")
                traceback.print_exc()
    
    def detect(self, frame):
        """Queue frame for detection."""
        if not self.detect_queue.full():
            self.detect_queue.put(frame)
    
    def get_last_result(self):
        """Get the latest detection result."""
        with self.lock:
            return self.detect_result
    
    def cleanup(self):
        """Stop the worker thread."""
        self.running = False
        if self.detect_thread.is_alive():
            self.detect_thread.join(timeout=1.0)


class HandTracker:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
    
    def is_pinch(self, hand_landmarks, frame_shape, threshold=0.05):
        """Check if thumb and index finger are pinching."""
        h, w, _ = frame_shape
        x1, y1 = int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h)
        x2, y2 = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)
        dist = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return (dist / w) < threshold
    
    def get_index_tip_coordinates(self, hand_landmarks, frame_shape):
        """Get coordinates of index fingertip."""
        h, w, _ = frame_shape
        cx = int(hand_landmarks.landmark[8].x * w)
        cy = int(hand_landmarks.landmark[8].y * h)
        return cx, cy
    
    def process_frame(self, frame):
        """Process a frame with MediaPipe Hands."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(frame_rgb)


class ObjectManipulator:
    def __init__(self):
        # Virtual objects with class + 3D model/thumbnail pairs
        self.virtual_objects = {
            "sofa": cv2.imread("sofa.jpg") if os.path.exists("sofa.jpg") else None,
            "chair": cv2.imread("chair.jpg") if os.path.exists("chair.jpg") else None,
            "tv": cv2.imread("tv.jpg") if os.path.exists("tv.jpg") else None,
            "bed": cv2.imread("bed.jpg") if os.path.exists("bed.jpg") else None
        }
        
        # Fill missing objects with colored rectangles
        for key in self.virtual_objects:
            if self.virtual_objects[key] is None:
                img = np.zeros((100, 150, 3), dtype=np.uint8)
                if key == "sofa":
                    color = (0, 0, 255)  # Red
                elif key == "chair":
                    color = (0, 255, 0)  # Green
                elif key == "tv":
                    color = (255, 0, 0)  # Blue
                else:
                    color = (255, 255, 0)  # Cyan
                
                cv2.rectangle(img, (10, 10), (140, 90), color, -1)
                cv2.putText(img, key, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                self.virtual_objects[key] = img
        
        # Detected objects (from YOLO)
        self.detected_objects = []
        
        # Moving object state
        self.moving_obj = None
        self.moving_class = None
        self.moving_pos = (0, 0)
        self.pick_offset = (0, 0)
        
        # Inpainting
        self.background = None
        self.mask = None
    
    def update_detected_objects(self, results):
        """Update detected objects from YOLO results."""
        if results is None:
            return
        
        # Process detection results
        try:
            self.detected_objects = []
            for det in results.xyxy[0]:  # Assuming batch size 1
                x1, y1, x2, y2, conf, cls = det
                cls_id = int(cls.item())
                cls_name = results.names[cls_id]
                # Record every detected object so we can draw boxes for all classes
                x1, y1, x2, y2 = map(int, [x1.item(), y1.item(), x2.item(), y2.item()])
                self.detected_objects.append({
                    'class': cls_name,
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf.item()
                })
        except Exception as e:
            print(f"Error processing detection results: {e}")
    
    def find_object_at_position(self, x, y):
        """Find detected object under fingertip."""
        for obj in self.detected_objects:
            x1, y1, x2, y2 = obj['bbox']
            if x1 <= x <= x2 and y1 <= y <= y2:
                return obj
        return None
    
    def grab_object(self, x, y, frame):
        """Try to grab an object at the given position."""
        # First see if we have a detected real object at this position
        obj = self.find_object_at_position(x, y)
        if obj:
            # We detected something real, use the virtual replacement
            class_name = obj['class']
            if class_name in self.virtual_objects:
                # Create background if needed
                if self.background is None:
                    self.background = frame.copy()
                
                # Get bounding box
                x1, y1, x2, y2 = obj['bbox']
                
                # Create mask for inpainting
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                self.mask = mask
                
                # Inpaint background
                self.background = cv2.inpaint(self.background, mask, 3, cv2.INPAINT_TELEA)
                
                # Set up virtual object
                self.moving_class = class_name
                self.moving_obj = self.virtual_objects[class_name].copy()
                
                # Start at detected position
                self.moving_pos = (x1, y1)
                # Calculate offset from grab point to top-left corner
                self.pick_offset = (x - x1, y - y1)
                
                return True
        
        # No detected object, try a virtual one instead
        if len(self.virtual_objects) > 0:
            # Just place a random virtual object
            class_name = list(self.virtual_objects.keys())[0]
            self.moving_class = class_name
            self.moving_obj = self.virtual_objects[class_name].copy()
            self.moving_pos = (x - self.moving_obj.shape[1]//2, y - self.moving_obj.shape[0]//2)
            self.pick_offset = (self.moving_obj.shape[1]//2, self.moving_obj.shape[0]//2)
            return True
        
        return False
    
    def move_object(self, x, y):
        """Move the grabbed object to a new position."""
        if self.moving_obj is None:
            return
        
        # Update position
        nx = x - self.pick_offset[0]
        ny = y - self.pick_offset[1]
        self.moving_pos = (nx, ny)
    
    def draw_objects(self, frame):
        """Draw detected and virtual objects on the frame."""
        display = frame.copy()
        
        # Apply background if exists
        if self.background is not None:
            display = self.background.copy()
        
        # Draw moving virtual object
        if self.moving_obj is not None:
            h, w = self.moving_obj.shape[:2]
            x, y = self.moving_pos
            
            # Ensure within frame
            h_frame, w_frame = frame.shape[:2]
            x = max(0, min(w_frame - w, x))
            y = max(0, min(h_frame - h, y))
            
            # Create region of interest
            roi = display[y:y+h, x:x+w]
            
            # Check if ROI is valid
            if roi.shape[:2] == self.moving_obj.shape[:2]:
                # Create mask from non-zero pixels
                gray = cv2.cvtColor(self.moving_obj, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)
                
                # Extract background from ROI using inverse mask
                bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                
                # Extract foreground from object using mask
                fg = cv2.bitwise_and(self.moving_obj, self.moving_obj, mask=mask)
                
                # Combine background + foreground
                dst = cv2.add(bg, fg)
                display[y:y+h, x:x+w] = dst
        
        # Draw bounding boxes for detected objects
        for obj in self.detected_objects:
            x1, y1, x2, y2 = obj['bbox']
            conf = obj['confidence']
            class_name = obj['class']
            
            # Draw box
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(display, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return display
    
    def release_object(self):
        """Place the object in the scene."""
        self.moving_obj = None
        self.moving_class = None


def main():
    # Initialize components
    detector = YOLODetector()
    hand_tracker = HandTracker()
    manipulator = ObjectManipulator()
    
    # Setup webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Error: cannot access webcam")
        return
    
    # FPS tracking
    fps_time = time.time()
    fps = 0
    
    # Main loop
    moving = False
    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate FPS
        current_time = time.time()
        frame_time = current_time - prev_time
        if frame_time > 0:
            fps = 0.9 * fps + 0.1 * (1 / frame_time)
        prev_time = current_time
        
        # Mirror image for more intuitive interaction
        frame = cv2.flip(frame, 1)
        
        # Run object detection in thread
        detector.detect(frame)
        results = detector.get_last_result()
        
        # Update detected objects list from YOLO
        if results is not None:
            manipulator.update_detected_objects(results)
        
        # Process hand landmarks
        hand_results = hand_tracker.process_frame(frame)
        
        # Hand interaction
        if hand_results.multi_hand_landmarks:
            # Draw landmarks
            for hand_landmarks in hand_results.multi_hand_landmarks:
                hand_tracker.mp_draw.draw_landmarks(frame, hand_landmarks, hand_tracker.mp_hands.HAND_CONNECTIONS)
            
            # Use first hand for interaction
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            cx, cy = hand_tracker.get_index_tip_coordinates(hand_landmarks, frame.shape)
            
            # Check for pinch (grab/release)
            if hand_tracker.is_pinch(hand_landmarks, frame.shape):
                if not moving:
                    # Try to grab an object
                    if manipulator.grab_object(cx, cy, frame):
                        moving = True
                else:
                    # Continue moving object
                    manipulator.move_object(cx, cy)
            else:
                # Release
                if moving:
                    manipulator.release_object()
                    moving = False
        
        # Draw objects
        display = manipulator.draw_objects(frame)
        
        # Draw FPS
        cv2.putText(display, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show
        cv2.imshow("AR with YOLO", display)
        
        # Handle exit
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # Esc or q
            break
    
    # Cleanup
    detector.cleanup()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main() 