import cv2
import time
import numpy as np
from collections import deque
import logging
from datetime import datetime
import threading
import queue
import torch
from queue import Empty

from gesture_detection import process_frame, draw_landmarks, get_index_tip_coordinates, is_index_only_up, is_pinch, is_fist, is_two_fingers_up, is_three_fingers_up, is_peace
from ui import draw_buttons, highlight_button, draw_thickness_slider, draw_hud, draw_help_overlay, button_height
from canvas import Canvas
from ar_yolo import YOLODetector, ObjectManipulator as YOLOObjectManipulator
from gan_inpainter import GANInpainter

# Check if CUDA is available for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DepthEstimator:
    def __init__(self):
        # Load MiDaS model for depth estimation
        try:
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
            self.depth_queue = queue.Queue(maxsize=1)
            self.depth_result = None
            self.depth_lock = threading.Lock()
            self.running = True
            
            # Start worker thread
            self.depth_thread = threading.Thread(target=self._depth_worker)
            self.depth_thread.daemon = True
            self.depth_thread.start()
            print("Depth estimation initialized successfully")
        except Exception as e:
            print(f"Error initializing depth estimation: {e}")
            self.midas = None
            self.running = False
    
    def _depth_worker(self):
        """Worker thread to compute depth maps asynchronously."""
        while self.running:
            try:
                img = self.depth_queue.get(timeout=0.1)
            except Empty:
                continue
            try:
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
        if not self.depth_queue.full() and self.running:
            self.depth_queue.put(img.copy())
    
    def get_last_depth_map(self):
        """Get the latest computed depth map."""
        with self.depth_lock:
            return self.depth_result
    
    def cleanup(self):
        """Stop the worker thread."""
        self.running = False
        if hasattr(self, 'depth_thread') and self.depth_thread.is_alive():
            self.depth_thread.join(timeout=1.0)

class ObjectManipulator:
    def __init__(self, inpainter=None):
        # Optional inpainting helper (GAN or OpenCV)
        self.inpainter = inpainter
        self.roi_img = None
        self.roi_mask = None
        self.roi_pos = (0, 0)
        self.roi_depth = 0
        self.moving = False
        self.pick_offset = (0, 0)
        self.background = None
        self.roi_img_scaled = None
        self.roi_mask_scaled = None
    
    def grab_object(self, frame, depth_map, cx, cy):
        """Segment object at position using edge detection and flood fill."""
        # Create background if needed
        if self.background is None:
            self.background = frame.copy()
        
        # Edge detection
        gray = cv2.cvtColor(self.background, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        inv_edges = cv2.bitwise_not(edges)
        
        # Clamp seed point to image bounds (prevent out-of-range error)
        h_e, w_e = inv_edges.shape
        cx_clamped = max(0, min(cx, w_e - 1))
        cy_clamped = max(0, min(cy, h_e - 1))
        
        # Flood fill from seed point
        mask = np.zeros((h_e + 2, w_e + 2), np.uint8)
        cv2.floodFill(inv_edges, mask, (cx_clamped, cy_clamped), 255)
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
        
        # Inpaint background (GAN-based if available, else OpenCV)
        if self.inpainter:
            self.background = self.inpainter.inpaint(self.background, seg_mask.astype(np.uint8))
        else:
            self.background = cv2.inpaint(self.background, seg_mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
        
        # Set position and offset
        self.roi_pos = (x0, y0)
        self.pick_offset = (cx - x0, cy - y0)
        self.moving = True
        self.roi_img_scaled = self.roi_img
        self.roi_mask_scaled = self.roi_mask
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
    
    def reset(self, frame):
        """Reset the manipulator with a new background frame."""
        self.background = frame.copy()
        self.moving = False
        self.roi_img = None

def get_index_tip_3d(hand_landmarks, depth_map, frame_shape):
    """Get 3D coordinates of index fingertip using depth map."""
    h, w, _ = frame_shape
    cx = int(hand_landmarks.landmark[8].x * w)
    cy = int(hand_landmarks.landmark[8].y * h)
    depth = 0
    if depth_map is not None:
        # Get depth at fingertip location
        if 0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1]:
            depth = depth_map[cy, cx]
    return cx, cy, depth

def draw_3d_strokes(frame, strokes_3d, depth_map):
    """Render recorded 3D strokes by scaling line thickness inversely to depth."""
    if depth_map is None:
        return frame
    disp = frame.copy()
    for stroke in strokes_3d:
        for x0, y0, z0, x1, y1, z1, color, thickness in stroke:
            avg_z = (z0 + z1) / 2.0
            # Closer strokes (lower z) appear thicker
            scale = 1.0 + (1.0 - avg_z / 255.0) * 2.0
            t2 = max(1, int(thickness * scale))
            cv2.line(disp, (x0, y0), (x1, y1), color, t2, lineType=cv2.LINE_AA)
    return disp

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger('HandDrawingApp')

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to access the webcam.")
        return

    # Initialize AR components: depth, GAN inpainter, edge/detector-based manipulators
    depth_estimator = DepthEstimator()
    gan_inpainter = GANInpainter()
    # Pass the GAN inpainting helper into the manipulator
    object_manipulator = ObjectManipulator(gan_inpainter)
    yolo_detector = YOLODetector()
    yolo_manipulator = YOLOObjectManipulator()
    depth_map = None
    depth_vis = None
    yolo_results = None
    
    # Initialize canvas and drawing state
    canvas = Canvas(frame.shape)
    # 3D stroke storage
    strokes_3d = []
    current_stroke_3d = None
    prev_depth = None
    current_color = (0, 0, 255)  # Red by default (BGR)
    line_thickness = 5
    min_thickness, max_thickness = 1, 30
    button_pressed = False
    prev_point = None
    button_labels = ["Clear", "Save", "Undo", "Redo", "Eraser", "Red", "Green", "Blue", "Yellow", "AR Mode", "AR Reset"]
    # Eraser mode state
    is_eraser_mode = False
    # Shape drawing state
    in_shape_mode = False
    shape_start = None
    shape_type = 'rectangle'  # Could toggle between rectangle, circle
    # Flood-fill and text annotation state
    flood_pressed = False
    text_mode = False
    text_buffer = ""
    text_anchor = None
    help_overlay = False  # Toggle help display
    ar_mode = False  # AR drawing/manipulation mode (off by default)
    # Smoothing buffer for drawing to reduce jitter
    smoothing_buffer = deque(maxlen=3)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # Mirror view
        
        # Depth estimation in background for AR
        if ar_mode:
            depth_estimator.get_depth(frame.copy())
            depth_map = depth_estimator.get_last_depth_map()
            if depth_map is not None:
                depth_vis = cv2.applyColorMap(depth_map, cv2.COLORMAP_PLASMA)
            # YOLO detection update
            yolo_detector.detect(frame.copy())
            yolo_results = yolo_detector.get_last_result()
            yolo_manipulator.update_detected_objects(yolo_results)
        
        # Synchronous hand detection per frame
        results = process_frame(frame)
        # Draw hand landmarks
        draw_landmarks(frame, results)

        # Calculate slider position at bottom each frame
        h, w, _ = frame.shape
        slider_x, slider_y = 10, h - 30
        slider_length, slider_thickness = w - 20, 5

        # Draw UI buttons and bottom thickness slider
        button_width = draw_buttons(frame, button_labels)
        slider_value = (line_thickness - min_thickness) / (max_thickness - min_thickness)
        draw_thickness_slider(frame, slider_x, slider_y, slider_length, slider_thickness, slider_value)
        # Draw HUD
        draw_hud(frame, is_eraser_mode, current_color, line_thickness)
        # Draw help overlay if enabled
        if help_overlay:
            draw_help_overlay(frame)
        
        # Show depth visualization in corner when in AR mode
        if ar_mode and depth_vis is not None:
            corner_h, corner_w = h//4, w//4
            depth_small = cv2.resize(depth_vis, (corner_w, corner_h))
            frame[0:corner_h, 0:corner_w] = depth_small

        # Handle text-mode input overlay
        if text_mode:
            h, w, _ = frame.shape
            # Draw input prompt
            cv2.rectangle(frame, (50, h//2 - 30), (w - 50, h//2 + 30), (0, 0, 0), -1)
            prompt = "Type text and press Enter: " + text_buffer
            cv2.putText(frame, prompt, (60, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Virtual Drawing", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter
                canvas.add_text(text_anchor, text_buffer, current_color, line_thickness)
                text_mode = False
            elif key != 255:
                text_buffer += chr(key)
            continue

        # Process hand gestures
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get 2D/3D fingertip coords depending on AR mode
                if ar_mode:
                    cx, cy, cz = get_index_tip_3d(hand_landmarks, depth_map, frame.shape)
                else:
                    cx, cy = get_index_tip_coordinates(hand_landmarks, frame.shape)
                    cz = None
                
                # YOLO-based grab: only on pinch & inside bounding box
                if ar_mode and is_pinch(hand_landmarks, frame.shape) and yolo_results is not None and yolo_manipulator.find_object_at_position(cx, cy):
                    if not hasattr(yolo_manipulator, 'moving_obj') or yolo_manipulator.moving_obj is None:
                        yolo_manipulator.grab_object(cx, cy, frame)
                    else:
                        yolo_manipulator.move_object(cx, cy)
                    continue
                # Edge-based grab: arbitrary object segmentation + GAN inpainting (only on pinch or reset)
                if ar_mode and (is_pinch(hand_landmarks, frame.shape) or is_fist(hand_landmarks)):
                    cx, cy, finger_depth = get_index_tip_3d(hand_landmarks, depth_map, frame.shape)
                    if is_pinch(hand_landmarks, frame.shape):
                        if not object_manipulator.moving:
                            object_manipulator.grab_object(frame, depth_map, cx, cy)
                        else:
                            object_manipulator.move_object(cx, cy, finger_depth)
                    else:
                        # Release on non-pinch
                        object_manipulator.release_object()
                    if is_fist(hand_landmarks):
                        object_manipulator.reset(frame)
                    continue
                # Regular drawing mode gestures
                # Flood-fill (three-finger)
                if is_three_fingers_up(hand_landmarks):
                    canvas.flood_fill((cx, cy), current_color)
                    continue
                # Text annotation (peace sign)
                if is_peace(hand_landmarks) and not text_mode:
                    text_mode = True
                    text_buffer = ''
                    text_anchor = (cx, cy)
                    continue
                # Shape drawing (two-finger)
                if is_two_fingers_up(hand_landmarks):
                    if shape_start is None:
                        shape_start = (cx, cy)
                    else:
                        x0, y0 = shape_start
                        if shape_type == 'rectangle':
                            pts = [((x0, y0),(cx, y0)),((cx, y0),(cx, cy)),((cx, cy),(x0, cy)),((x0, cy),(x0, y0))]
                            for p0, p1 in pts:
                                canvas.add_segment(p0, p1, current_color, line_thickness)
                        elif shape_type == 'circle':
                            radius = int(((cx-x0)**2 + (cy-y0)**2)**0.5)
                            cv2.circle(canvas.canvas, shape_start, radius, current_color, line_thickness)
                        shape_start = None
                    prev_point = None
                    continue
                # Button area
                if cy < button_height:
                    idx = cx // button_width
                    highlight_button(frame, idx, button_labels)
                    if not button_pressed:
                        # Map button index to actions
                        if idx == 0: canvas.clear()
                        elif idx == 1: canvas.save()
                        elif idx == 2: canvas.undo()
                        elif idx == 3: canvas.redo()
                        elif idx == 4: current_color = (0,0,0)
                        elif idx == 5: current_color = (0,0,255)
                        elif idx == 6: current_color = (0,255,0)
                        elif idx == 7: current_color = (255,0,0)
                        elif idx == 8: current_color = (0,255,255)
                        elif idx == 9: ar_mode = not ar_mode
                        elif idx == 10: 
                            if ar_mode:
                                object_manipulator.reset(frame)
                        button_pressed = True
                    prev_point = None
                    continue
                button_pressed = False
                # Slider area
                if abs(cy - slider_y) < slider_thickness*3:
                    ratio = (cx - slider_x)/slider_length
                    ratio = max(0.0, min(1.0, ratio))
                    line_thickness = int(min_thickness + ratio*(max_thickness-min_thickness))
                    prev_point = None
                    continue
                # Regular freehand drawing and erasing
                erasing = is_pinch(hand_landmarks, frame.shape)
                if is_index_only_up(hand_landmarks) or erasing:
                    if prev_point is None:
                        # Start a new 2D stroke and corresponding 3D stroke
                        canvas.start_stroke()
                        current_stroke_3d = []
                        prev_point = (cx, cy)
                        prev_depth = cz
                    color = (0, 0, 0) if erasing else current_color
                    # Record on 2D canvas
                    canvas.add_segment(prev_point, (cx, cy), color, line_thickness)
                    # Record 3D segment if depth available
                    if cz is not None and prev_depth is not None:
                        current_stroke_3d.append((prev_point[0], prev_point[1], prev_depth,
                                                  cx, cy, cz, color, line_thickness))
                    prev_point = (cx, cy)
                    prev_depth = cz
                    continue
                # End stroke when gesture ends
                if prev_point is not None:
                    canvas.end_stroke()
                    # Commit 3D stroke
                    if current_stroke_3d:
                        strokes_3d.append(current_stroke_3d)
                        current_stroke_3d = None
                prev_point = None

        # Prepare display frame
        if ar_mode:
            # Use manipulated background or live frame
            if object_manipulator.background is not None and not object_manipulator.moving:
                display = object_manipulator.background.copy()
            else:
                display = frame.copy()
            # Draw arbitrary object and YOLO-based virtual objects
            display = object_manipulator.draw_object(display)
            display = yolo_manipulator.draw_objects(display)
            # Overlay user drawings in 3D
            display = draw_3d_strokes(display, strokes_3d, depth_map)
        else:
            # Overlay canvas onto the live frame
            display = cv2.add(frame, canvas.get_canvas())
        
        # Show the result
        cv2.imshow("Virtual Drawing", display)

        # Handle keyboard shortcuts
        key = cv2.waitKey(1) & 0xFF
        if key == ord('u'):
            canvas.undo()
        elif key == ord('r'):
            canvas.redo()
        elif key == ord('c'):
            canvas.clear()
            logger.info('Clear canvas via keypress')
        elif key == ord('h'):
            help_overlay = not help_overlay
            logger.info(f'Help overlay toggled to {help_overlay}')
        elif key == ord('a'):
            ar_mode = not ar_mode
            logger.info(f'AR mode toggled to {ar_mode}')
        elif key == ord('q'):
            break

    cap.release()
    # No async threads to shut down for hand detection
    logger.info('Application exiting')
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main() 