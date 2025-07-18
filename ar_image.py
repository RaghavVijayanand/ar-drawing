import sys
import cv2
import mediapipe as mp
from math import sqrt
import numpy as np

def is_pinch(hand_landmarks, frame_shape, threshold=0.05):
    h, w, _ = frame_shape
    x1, y1 = int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h)
    x2, y2 = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)
    dist = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return (dist / w) < threshold


def main():
    if len(sys.argv) < 2:
        print("Usage: python ar_image.py <path_to_image>")
        return
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: cannot load image at {img_path}")
        return
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

    # Setup camera
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Error: cannot access webcam")
        return
    h_f, w_f, _ = frame.shape
    h_i, w_i = img.shape[:2]
    # Background holds the inpainted scene
    background = img.copy()
    # No ROI initially
    roi_img = None
    roi_mask = None
    roi_pos = (0, 0)
    moving = False
    pick_offset = (0, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame_h, frame_w, _ = frame.shape

        # Process hand
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            # Tip coordinates
            cx = int(hand_landmarks.landmark[8].x * frame_w)
            cy = int(hand_landmarks.landmark[8].y * frame_h)
            # Pinch detection
            if is_pinch(hand_landmarks, frame.shape):
                if not moving:
                    # First pinch: segment object under finger
                    # Edge detection
                    gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    inv = cv2.bitwise_not(edges)
                    # Flood fill mask
                    mask = np.zeros((h_i+2, w_i+2), np.uint8)
                    fx, fy = cx, cy
                    cv2.floodFill(inv, mask, (fx, fy), 255)
                    seg = mask[1:-1,1:-1]
                    # Extract ROI
                    ys, xs = np.where(seg>0)
                    if ys.size and xs.size:
                        y0, y1 = ys.min(), ys.max()
                        x0, x1 = xs.min(), xs.max()
                        roi_img = background[y0:y1+1, x0:x1+1].copy()
                        roi_mask = seg[y0:y1+1, x0:x1+1].astype(np.uint8)*255
                        # Inpaint background
                        background = cv2.inpaint(background, seg.astype(np.uint8), 3, cv2.INPAINT_TELEA)
                        # Initialize moving
                        moving = True
                        pick_offset = (cx - x0, cy - y0)
                        roi_pos = (x0, y0)
                else:
                    # Continue moving ROI
                    nx = cx - pick_offset[0]
                    ny = cy - pick_offset[1]
                    roi_pos = (np.clip(nx,0,frame_w - roi_img.shape[1]), np.clip(ny,0, frame_h - roi_img.shape[0]))
                # Skip drawing background overlay until after segmentation/move
                continue
            else:
                moving = False
        # Compose frame: show inpainted background
        overlay = frame.copy()
        # Resize background to fit frame
        base = cv2.resize(background, (frame_w, frame_h))
        overlay = base.copy()
        # Draw moving ROI if exists
        if roi_img is not None and moving:
            x0, y0 = roi_pos
            h_r, w_r = roi_img.shape[:2]
            overlay[y0:y0+h_r, x0:x0+w_r] = roi_img

        cv2.imshow('AR Image Manipulation', overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # Esc or q
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main() 