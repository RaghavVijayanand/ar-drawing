import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """
    Calculate the angle between vectors AB and BC.
    Args:
    a, b, c: Coordinates of points A, B, C as (x, y, z).

    Returns:
    Angle in degrees.
    """
    ab = np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
    bc = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]])
    
    # Normalize vectors to avoid precision issues
    norm_ab = np.linalg.norm(ab)
    norm_bc = np.linalg.norm(bc)
    if norm_ab == 0 or norm_bc == 0:  # Prevent division by zero
        return 0.0
    
    ab_normalized = ab / norm_ab
    bc_normalized = bc / norm_bc
    
    # Compute dot product and angle
    dot_product = np.dot(ab_normalized, bc_normalized)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Clip to avoid floating-point errors
    angle = np.arccos(dot_product)
    return np.degrees(angle)

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty frame.")
        continue

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Convert back to BGR for visualization
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract coordinates and scale them to image size
            img_height, img_width, _ = image.shape
            landmarks = [(lm.x * img_width, lm.y * img_height, lm.z * img_width) 
                         for lm in hand_landmarks.landmark]
            
            # Calculate angles for all joints
            angles = {}
            
            # Thumb
            angles["Thumb_CMC"] = calculate_angle(landmarks[0], landmarks[1], landmarks[2])
            angles["Thumb_MCP"] = calculate_angle(landmarks[1], landmarks[2], landmarks[3])
            angles["Thumb_IP"] = calculate_angle(landmarks[2], landmarks[3], landmarks[4])
            
            # Fingers (Index, Middle, Ring, Pinky)
            fingers = {
                "Index": [5, 6, 7, 8],
                "Middle": [9, 10, 11, 12],
                "Ring": [13, 14, 15, 16],
                "Pinky": [17, 18, 19, 20],
            }
            
            for finger, points in fingers.items():
                angles[f"{finger}_MCP"] = calculate_angle(landmarks[0], landmarks[points[0]], landmarks[points[1]])
                angles[f"{finger}_PIP"] = calculate_angle(landmarks[points[0]], landmarks[points[1]], landmarks[points[2]])
                angles[f"{finger}_DIP"] = calculate_angle(landmarks[points[1]], landmarks[points[2]], landmarks[points[3]])
            
            # Display angles on the video feed
            y_offset = 30
            for joint, angle in angles.items():
                cv2.putText(image, f"{joint}: {int(angle)}", 
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_offset += 20

    # Display the processed video feed
    cv2.imshow('Hand Joint Angles', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    ab = np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
    bc = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]])
    norm_ab = np.linalg.norm(ab)
    norm_bc = np.linalg.norm(bc)
    if norm_ab == 0 or norm_bc == 0:
        return 0.0
    ab_normalized = ab / norm_ab
    bc_normalized = bc / norm_bc
    dot_product = np.dot(ab_normalized, bc_normalized)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle = np.arccos(dot_product)
    return np.degrees(angle)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            img_height, img_width, _ = image.shape
            landmarks = [(lm.x * img_width, lm.y * img_height, lm.z * img_width) 
                         for lm in hand_landmarks.landmark]
            
            angles = {}

            # Thumb
            angles["Thumb_CMC"] = calculate_angle(landmarks[0], landmarks[1], landmarks[2])
            angles["Thumb_MCP"] = calculate_angle(landmarks[1], landmarks[2], landmarks[3])
            angles["Thumb_IP"] = calculate_angle(landmarks[2], landmarks[3], landmarks[4])

            # Fingers
            fingers = {
                "Index": [5, 6, 7, 8],
                "Middle": [9, 10, 11, 12],
                "Ring": [13, 14, 15, 16],
                "Pinky": [17, 18, 19, 20],
            }

            for finger, points in fingers.items():
                angles[f"{finger}_MCP"] = calculate_angle(landmarks[0], landmarks[points[0]], landmarks[points[1]])
                angles[f"{finger}_PIP"] = calculate_angle(landmarks[points[0]], landmarks[points[1]], landmarks[points[2]])
                angles[f"{finger}_DIP"] = calculate_angle(landmarks[points[1]], landmarks[points[2]], landmarks[points[3]])
            
            # Display angles
            y_offset = 30
            for joint, angle in angles.items():
                cv2.putText(image, f"{joint}: {int(angle)}", (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_offset += 20

    cv2.imshow('Hand Angles', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
