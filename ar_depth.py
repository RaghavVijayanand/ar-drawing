import cv2
import numpy as np
import mediapipe as mp
import torch
import os
import sys
from math import sqrt
from threading import Thread, Lock
from queue import Queue

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DepthEstimator:
    def __init__(self):
        # Load MiDaS model for depth estimation
        midas_model_type = "MiDaS_small"  # or DPT_Large
        self.midas = torch.hub.load("intel-isl/MiDaS", midas_model_type)
        self.midas.to(device)
        self.midas.eval()
        
        # Input transformation
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if midas_model_type == "DPT_Large" or midas_model_type == "DPT_Hybrid":
            self.transform = self.midas_transforms.dpt_transform
        else:
            self.transform = self.midas_transforms.small_transform
        
        # Depth estimation thread
        self.depth_queue = Queue(maxsize=1)
        self.depth_result = None
        self.depth_lock = Lock()
        self.running = True
        
        # Start worker thread
        self.depth_thread = Thread(target=self._depth_worker)
        self.depth_thread.daemon = True
        self.depth_thread.start()
    
    def _depth_worker(self):
        """Worker thread to compute depth maps asynchronously."""
        while self.running:
            try:
                img = self.depth_queue.get(timeout=0.1)
                # Transform input for model
                input_batch = self.transform(img).to(device)
                
                # Prediction and resize to original size
                with torch.no_grad():
                    prediction = self.midas(input_batch)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=img.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()
                
                depth_map = prediction.cpu().numpy()
                
                # Normalize depth map
                depth_min = depth_map.min()
                depth_max = depth_map.max()
                if depth_max - depth_min > 0:
                    depth_map = 255 * (depth_map - depth_min) / (depth_max - depth_min)
                    depth_map = depth_map.astype(np.uint8)
                
                # Store result
                with self.depth_lock:
                    self.depth_result = depth_map
            except Exception as e:
                print(f"Depth estimation error: {e}")
    
    def get_depth(self, img):
        """Queue image for depth estimation."""
        if not self.depth_queue.full():
            self.depth_queue.put(img.copy())
    
    def get_last_depth_map(self):
        """Get the latest computed depth map."""
        with self.depth_lock:
            return self.depth_result
    
    def cleanup(self):
        """Stop the worker thread."""
        self.running = False
        if self.depth_thread.is_alive():
            self.depth_thread.join(timeout=1.0)


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
    
    def is_fist(self, hand_landmarks):
        """Check if hand is making a fist."""
        # Check if all four fingers are down
        idx_down = hand_landmarks.landmark[8].y > hand_landmarks.landmark[6].y
        mid_down = hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y
        ring_down = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
        pinky_down = hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y
        return idx_down and mid_down and ring_down and pinky_down
    
    def get_index_tip_coordinates(self, hand_landmarks, frame_shape):
        """Get coordinates of index fingertip."""
        h, w, _ = frame_shape
        cx = int(hand_landmarks.landmark[8].x * w)
        cy = int(hand_landmarks.landmark[8].y * h)
        return cx, cy
    
    def get_index_tip_3d(self, hand_landmarks, depth_map, frame_shape):
        """Get 3D coordinates of index fingertip using depth map."""
        cx, cy = self.get_index_tip_coordinates(hand_landmarks, frame_shape)
        depth = 0
        if depth_map is not None:
            # Get depth at fingertip location
            if 0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1]:
                depth = depth_map[cy, cx]
        return cx, cy, depth
    
    def process_frame(self, frame):
        """Process a frame with MediaPipe Hands."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(frame_rgb)


class ObjectManipulator:
    def __init__(self):
        self.roi_img = None
        self.roi_mask = None
        self.roi_pos = (0, 0)
        self.roi_depth = 0
        self.moving = False
        self.pick_offset = (0, 0)
        self.background = None
    
    def grab_object(self, frame, depth_map, cx, cy):
        """Segment object at position using edge detection and flood fill."""
        # Create background if needed
        if self.background is None:
            self.background = frame.copy()
        
        # Edge detection
        gray = cv2.cvtColor(self.background, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        inv_edges = cv2.bitwise_not(edges)
        
        # Flood fill from point
        h, w = inv_edges.shape
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(inv_edges, mask, (cx, cy), 255)
        seg_mask = mask[1:-1, 1:-1]
        
        # Find bounding box of segmented region
        ys, xs = np.where(seg_mask > 0)
        if len(ys) == 0 or len(xs) == 0:
            return False
        
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        
        # Extract ROI and mask
        self.roi_img = self.background[y0:y1+1, x0:x1+1].copy()
        self.roi_mask = seg_mask[y0:y1+1, x0:x1+1].astype(np.uint8) * 255
        
        # Get average depth of object
        if depth_map is not None:
            object_depths = depth_map[y0:y1+1, x0:x1+1][self.roi_mask > 0]
            if len(object_depths) > 0:
                self.roi_depth = np.median(object_depths)
        
        # Inpaint background
        self.background = cv2.inpaint(self.background, seg_mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
        
        # Set position and offset
        self.roi_pos = (x0, y0)
        self.pick_offset = (cx - x0, cy - y0)
        self.moving = True
        return True
    
    def move_object(self, cx, cy, depth):
        """Move the grabbed object to new position, with depth scaling."""
        if not self.moving or self.roi_img is None:
            return
        
        # Scale object based on relative depth
        scale_factor = 1.0
        if self.roi_depth > 0 and depth > 0:
            scale_factor = self.roi_depth / max(1, depth)
        
        if scale_factor < 0.5:
            scale_factor = 0.5
        elif scale_factor > 2.0:
            scale_factor = 2.0
        
        # Update position (accounting for scaling)
        nx = cx - int(self.pick_offset[0] * scale_factor)
        ny = cy - int(self.pick_offset[1] * scale_factor)
        self.roi_pos = (nx, ny)
        
        # Update displayed object size if depth changes significantly
        if abs(scale_factor - 1.0) > 0.1:
            h, w = self.roi_img.shape[:2]
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            if new_h > 0 and new_w > 0:
                self.roi_img_scaled = cv2.resize(self.roi_img, (new_w, new_h))
                self.roi_mask_scaled = cv2.resize(self.roi_mask, (new_w, new_h))
            else:
                self.roi_img_scaled = self.roi_img
                self.roi_mask_scaled = self.roi_mask
        else:
            self.roi_img_scaled = self.roi_img
            self.roi_mask_scaled = self.roi_mask
    
    def draw_object(self, frame):
        """Draw the object on the frame at current position."""
        if not self.moving or self.roi_img is None:
            return frame
        
        # Get scaled ROI dimensions
        h_r, w_r = self.roi_img_scaled.shape[:2]
        x0, y0 = self.roi_pos
        
        # Ensure position is within frame
        h, w = frame.shape[:2]
        x0 = max(0, min(w - w_r, x0))
        y0 = max(0, min(h - h_r, y0))
        
        # Create mask for transparent blending
        roi_area = frame[y0:y0+h_r, x0:x0+w_r]
        if roi_area.shape[:2] != self.roi_mask_scaled.shape:
            return frame
        
        # Create mask in 3 channels
        mask_3ch = cv2.merge([self.roi_mask_scaled, self.roi_mask_scaled, self.roi_mask_scaled]) / 255.0
        
        # Overlay ROI on frame
        frame[y0:y0+h_r, x0:x0+w_r] = roi_area * (1 - mask_3ch) + self.roi_img_scaled * mask_3ch
        
        return frame
    
    def release_object(self):
        """Release the grabbed object."""
        self.moving = False


def main():
    # Check for image path
    if len(sys.argv) < 2:
        print("Usage: python ar_depth.py <path_to_image>")
        print("Running in camera-only mode...")
        use_image = False
    else:
        img_path = sys.argv[1]
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: cannot load image at {img_path}")
            return
        use_image = True
    
    # Initialize components
    depth_estimator = DepthEstimator()
    hand_tracker = HandTracker()
    manipulator = ObjectManipulator()
    
    # Setup webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Error: cannot access webcam")
        return
    
    # If using an image, set as background
    if use_image:
        img_resized = cv2.resize(img, (frame.shape[1], frame.shape[0]))
        manipulator.background = img_resized
    
    # Main loop
    depth_vis = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # Mirror for intuitive interaction
        
        # Process depth on background or frame
        depth_estimator.get_depth(manipulator.background if manipulator.background is not None else frame)
        depth_map = depth_estimator.get_last_depth_map()
        
        # Create visualization of depth
        if depth_map is not None:
            depth_vis = cv2.applyColorMap(depth_map, cv2.COLORMAP_PLASMA)
        
        # Process hand landmarks
        results = hand_tracker.process_frame(frame)
        if results.multi_hand_landmarks:
            # Draw landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                hand_tracker.mp_draw.draw_landmarks(frame, hand_landmarks, hand_tracker.mp_hands.HAND_CONNECTIONS)
            
            # Use first hand for interaction
            hand_landmarks = results.multi_hand_landmarks[0]
            cx, cy = hand_tracker.get_index_tip_coordinates(hand_landmarks, frame.shape)
            
            # Get 3D coordinates with depth
            cx, cy, depth = hand_tracker.get_index_tip_3d(hand_landmarks, depth_map, frame.shape)
            
            # Check for pinch (grab/release)
            if hand_tracker.is_pinch(hand_landmarks, frame.shape):
                if not manipulator.moving:
                    manipulator.grab_object(frame, depth_map, cx, cy)
                else:
                    manipulator.move_object(cx, cy, depth)
            else:
                manipulator.release_object()
            
            # Check for fist (reset scene)
            if hand_tracker.is_fist(hand_landmarks):
                manipulator.background = frame.copy()
                manipulator.moving = False
        
        # Apply background if exists
        if manipulator.background is not None and not manipulator.moving:
            display = manipulator.background.copy()
        else:
            display = frame.copy()
        
        # Draw object if moving
        display = manipulator.draw_object(display)
        
        # Show depth visualization in corner
        if depth_vis is not None:
            h, w = frame.shape[:2]
            depth_small = cv2.resize(depth_vis, (w//4, h//4))
            display[0:h//4, 0:w//4] = depth_small
        
        # Display
        cv2.imshow("AR with Depth", display)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # Esc or q
            break
    
    # Cleanup
    depth_estimator.cleanup()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main() 