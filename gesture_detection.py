import cv2
import mediapipe as mp
from math import sqrt

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


def process_frame(frame):
    """Process the frame through MediaPipe and return the results."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    return results


def draw_landmarks(frame, results):
    """Draw detected hand landmarks on the frame."""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


def get_index_tip_coordinates(hand_landmarks, frame_shape):
    """Get (x, y) coordinates of the index fingertip (landmark 8)."""
    h, w, _ = frame_shape
    cx = int(hand_landmarks.landmark[8].x * w)
    cy = int(hand_landmarks.landmark[8].y * h)
    return cx, cy


def is_index_only_up(hand_landmarks):
    """Return True if only the index finger is up (not the middle finger)."""
    index_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
    middle_up = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
    return index_up and not middle_up


def is_pinch(hand_landmarks, frame_shape, threshold=0.15):
    """Return True if thumb and index fingertip are pinched (close)."""
    h, w, _ = frame_shape
    x1, y1 = int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h)
    x2, y2 = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)
    dist = sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return dist / w < threshold


def is_fist(hand_landmarks):
    """Return True if all four fingers (index, middle, ring, pinky) are folded (fist)."""
    # Each fingertip landmark y > corresponding pip y indicates finger down
    idx_down = hand_landmarks.landmark[8].y > hand_landmarks.landmark[6].y
    mid_down = hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y
    ring_down = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
    pinky_down = hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y
    # Thumb folds sideways; drop strict thumb requirement for fist
    return idx_down and mid_down and ring_down and pinky_down


def is_two_fingers_up(hand_landmarks):
    """Return True if only the index and middle fingers are up (for shape drawing)."""
    index_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
    middle_up = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
    ring_up = hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y
    pinky_up = hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y
    return index_up and middle_up and not ring_up and not pinky_up


def is_three_fingers_up(hand_landmarks):
    """Return True if index, middle, and ring fingers are up (for flood-fill)."""
    index_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
    middle_up = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
    ring_up = hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y
    pinky_up = hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y
    return index_up and middle_up and ring_up and not pinky_up


def is_peace(hand_landmarks):
    """Return True if index and middle fingers are up and ring and pinky down and thumb down (peace sign for text)."""
    # Index and middle up
    index_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
    middle_up = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
    # Ring, pinky, and thumb down
    ring_down = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
    pinky_down = hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y
    thumb_down = hand_landmarks.landmark[4].y > hand_landmarks.landmark[3].y
    return index_up and middle_up and ring_down and pinky_down and thumb_down 